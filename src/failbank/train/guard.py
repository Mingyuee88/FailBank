# Derived from VLA-Arena's OpenPI training script (vla_arena/models/openpi/scripts/train.py,
# Apache-2.0, Copyright 2025 The VLA-Arena Authors), with the record-weighted objective and
# the held-out guard added.
"""Stage 4: the weighted LoRA objective and the held-out update guard.

Objective. The flow-matching loss of the policy is reduced over the FIRST action of each
chunk only, weighted per record:

    L = sum_b w_b * m_b * l_b[0] / max(sum_b w_b * m_b, 1)

where ``w_b`` is the record weight (teacher weight for triggered records, ``quiet_weight``
for quiet ones, 0 otherwise) and ``m_b = [w_b > 0]``. For triggered records the chunk's
first action has been replaced by the teacher target (see ``derived_lora_data``).

Guard. Before training, the loaded starting checkpoint is evaluated once on a fixed batch of
the held-out split (reference statistics). At every validation point the candidate is
accepted only if

    quiet_flow_ratio  = flow_loss(candidate) / flow_loss(start)  <= 1.10
    quiet_action_drift = mean|a0(candidate) - a0(start)|          <= 0.05

and the best accepted candidate by weighted held-out loss is kept. If no validation point
passes, nothing is saved.

Naming trap: "quiet" in ``quiet_flow_ratio`` / ``quiet_action_drift`` is historical. Both are
computed on the WHOLE fixed validation batch (triggered and quiet records alike), with an
unweighted full-horizon mean for the flow loss. The reference is the checkpoint the run
STARTED from (``base_state``), which equals the original base only when training starts
from base -- as every paper run does.
"""
from __future__ import annotations

import dataclasses
import functools
import logging

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import vla_arena.models.openpi.src.openpi.models.model as _model
import vla_arena.models.openpi.src.openpi.shared.array_typing as at
import vla_arena.models.openpi.src.openpi.shared.nnx_utils as nnx_utils
import vla_arena.models.openpi.src.openpi.training.sharding as sharding
import vla_arena.models.openpi.src.openpi.training.utils as training_utils

from failbank.train._openpi import upstream_train_module


@at.typecheck
def weighted_masked_first_action_mean(
    chunked_loss, include_mask, per_record_weight, first_k=1
):
    """Weighted GLOBAL mean over the first ``first_k`` action-horizon steps.

    Training is jax.jit + FSDP (no pmap), so a plain global sum/sum is correct -- XLA
    inserts the all-reduce for the replicated scalar output. ``first_k`` is static; the
    paper uses 1 (first action only).
    """
    chunked_loss = jnp.asarray(chunked_loss)
    if chunked_loss.ndim < 2:
        raise ValueError(
            'Expected chunked_loss with shape [B, action_horizon, ...], got '
            f'{chunked_loss.shape}'
        )
    if chunked_loss.ndim > 2:
        chunked_loss = jnp.mean(
            chunked_loss, axis=tuple(range(2, chunked_loss.ndim))
        )
    include_mask = jnp.asarray(include_mask, dtype=chunked_loss.dtype)
    per_record_weight = jnp.asarray(per_record_weight, dtype=chunked_loss.dtype)
    record_mask = include_mask * per_record_weight
    _k = int(first_k or 1)
    if _k <= 1:
        first_action_onehot = jax.nn.one_hot(
            0, chunked_loss.shape[1], dtype=chunked_loss.dtype
        )
    else:
        _k = min(_k, chunked_loss.shape[1])
        first_action_onehot = jnp.concatenate([
            jnp.ones((_k,), dtype=chunked_loss.dtype),
            jnp.zeros((chunked_loss.shape[1] - _k,), dtype=chunked_loss.dtype),
        ])
    weighted_mask = record_mask[:, None] * first_action_onehot[None, :]
    numerator = jnp.sum(chunked_loss * weighted_mask)
    denominator = jnp.maximum(jnp.sum(weighted_mask), 1.0)
    return numerator / denominator


def weighted_train_step(
    config,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple,
    first_k: int = 1,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """OpenPI's train_step with the record-weighted objective; batch is a 4-tuple."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        include_mask: at.Array,
        per_record_weight: at.Array,
    ):
        chunked_loss = model.compute_loss(
            rng, observation, actions, train=True
        )
        return weighted_masked_first_action_mean(
            chunked_loss, include_mask, per_record_weight, first_k
        )

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions, include_mask, per_record_weight = batch

    # Filter out frozen params. (The hard "batch has a positive-weight record" check lives
    # in the Python data loader; it cannot raise on traced values.)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, actions, include_mask, per_record_weight
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_params, opt_state=new_opt_state
    )
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old
                + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(
                nnx_utils.PathRegex(
                    '.*/(bias|scale|pos_embedding|input_embedding)'
                )
            ),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        'loss': loss,
        'grad_norm': optax.global_norm(grads),
        'param_norm': optax.global_norm(kernel_params),
    }
    return new_state, info


def train_with_heldout_guard(
    config,
    *,
    base_checkpoint,
    train_loader_factory,
    validation_loader_factory,
    validation_interval=25,
    patience=4,
    quiet_flow_loss_ratio_limit=1.10,
    quiet_action_drift_limit=0.05,
    snapshot_callback=None,
    first_k=1,
):
    """Train one held-out fold; retain the best guard-accepted state.

    Initialises fresh from ``config.weight_loader`` (the starting checkpoint), never
    resumes, and never saves: the caller saves an adapter only when
    ``metrics['accepted']`` is True.
    """
    if validation_interval <= 0:
        raise ValueError('validation_interval must be positive')
    if patience <= 0:
        raise ValueError('patience must be positive')
    init_train_state = upstream_train_module().init_train_state

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    init_rng = jax.random.key(42)
    train_rng = jax.random.key(43)
    fixed_eval_rng = jax.random.key(0)

    with sharding.set_mesh(mesh):
        train_state, train_state_sharding = init_train_state(
            config, init_rng, mesh, resume=False
        )

        ptrain_step = jax.jit(
            functools.partial(weighted_train_step, config, first_k=first_k),
            in_shardings=(
                replicated_sharding,
                train_state_sharding,
                data_sharding,
            ),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

        def eval_step(rng, state, batch):
            model = nnx.merge(state.model_def, state.params)
            model.eval()
            observation, actions, include_mask, per_record_weight = batch
            chunked_loss = model.compute_loss(
                rng, observation, actions, train=False
            )
            triggered_loss = weighted_masked_first_action_mean(
                chunked_loss, include_mask, per_record_weight, first_k
            )
            # Full-horizon flow loss on the held-out batch as a drift proxy.
            quiet_flow_loss = jnp.mean(chunked_loss)
            # First-action drift of the sampled policy against the starting checkpoint.
            sampled_actions = model.sample_actions(rng, observation)
            quiet_first_action = sampled_actions[:, 0]
            return triggered_loss, quiet_flow_loss, quiet_first_action

        peval_step = jax.jit(
            eval_step,
            in_shardings=(
                replicated_sharding,
                train_state_sharding,
                data_sharding,
            ),
            out_shardings=(
                replicated_sharding,
                replicated_sharding,
                replicated_sharding,
            ),
        )

        # ptrain_step donates the live train state; keep a separate physical
        # snapshot for the reference metrics and for the best accepted state.
        pcopy_state = jax.jit(
            lambda state: jax.tree.map(jnp.copy, state),
            in_shardings=(train_state_sharding,),
            out_shardings=train_state_sharding,
        )

        base_state = pcopy_state(train_state)
        jax.block_until_ready(base_state)

        validation_loader = validation_loader_factory(data_sharding)
        validation_iterator = iter(validation_loader)
        try:
            validation_batch = next(validation_iterator)
        except StopIteration as exc:
            raise ValueError(
                'validation_loader_factory returned an empty loader'
            ) from exc

        (
            _val_obs,
            _val_actions,
            validation_include_mask,
            validation_record_weight,
        ) = validation_batch
        validation_positive_mass = float(
            np.asarray(
                jax.device_get(
                    jnp.sum(
                        jnp.asarray(validation_include_mask)
                        * jnp.asarray(validation_record_weight)
                    )
                )
            )
        )
        if validation_positive_mass <= 0:
            raise ValueError(
                'The fixed validation batch has no included record with '
                'positive teacher weight'
            )

        (
            base_triggered_loss_d,
            base_quiet_flow_loss_d,
            base_quiet_first_action,
        ) = peval_step(fixed_eval_rng, base_state, validation_batch)
        jax.block_until_ready(
            (base_triggered_loss_d, base_quiet_flow_loss_d, base_quiet_first_action)
        )
        base_triggered_loss = float(np.asarray(jax.device_get(base_triggered_loss_d)))
        base_quiet_flow_loss = float(np.asarray(jax.device_get(base_quiet_flow_loss_d)))

        best_state = base_state
        best_accepted = False
        best_triggered_loss = float('inf')
        best_quiet_flow_ratio = None
        best_quiet_action_drift = None
        best_step = None
        validations_run = 0
        rejected_validations = 0
        consecutive_non_improvements = 0
        stopped_early = False

        train_loader = train_loader_factory(data_sharding)
        train_iterator = iter(train_loader)

        logging.info(
            'Update init from %s; start triggered=%.6f quiet_flow=%.6f',
            base_checkpoint, base_triggered_loss, base_quiet_flow_loss,
        )

        for loop_step in range(config.num_train_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_loader = train_loader_factory(data_sharding)
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            train_state, info = ptrain_step(train_rng, train_state, batch)

            completed_step = loop_step + 1
            should_validate = (
                completed_step % validation_interval == 0
                or completed_step == config.num_train_steps
            )
            if not should_validate:
                continue

            (
                cur_triggered_d,
                cur_quiet_flow_d,
                cur_quiet_first_action,
            ) = peval_step(fixed_eval_rng, train_state, validation_batch)
            jax.block_until_ready(
                (cur_triggered_d, cur_quiet_flow_d, cur_quiet_first_action)
            )
            current_triggered_loss = float(np.asarray(jax.device_get(cur_triggered_d)))
            current_quiet_flow_loss = float(np.asarray(jax.device_get(cur_quiet_flow_d)))
            cur_qfa_host = np.asarray(jax.device_get(cur_quiet_first_action))
            base_qfa_host = np.asarray(jax.device_get(base_quiet_first_action))

            quiet_flow_denominator = max(
                base_quiet_flow_loss, float(np.finfo(np.float32).tiny)
            )
            quiet_flow_ratio = current_quiet_flow_loss / quiet_flow_denominator
            quiet_action_drift = float(np.mean(np.abs(cur_qfa_host - base_qfa_host)))

            accepted = (
                np.isfinite(current_triggered_loss)
                and np.isfinite(quiet_flow_ratio)
                and np.isfinite(quiet_action_drift)
                and quiet_flow_ratio <= quiet_flow_loss_ratio_limit
                and quiet_action_drift <= quiet_action_drift_limit
            )
            improved = accepted and current_triggered_loss < best_triggered_loss
            validations_run += 1

            print(
                "VALIDATION_POINT step=%s triggered_loss=%.6f quiet_flow_ratio=%.5f "
                "quiet_drift=%.6f limits(flow=%.3f drift=%.3f) accepted=%s improved=%s"
                % (completed_step, float(current_triggered_loss), float(quiet_flow_ratio),
                   float(quiet_action_drift), float(quiet_flow_loss_ratio_limit),
                   float(quiet_action_drift_limit), accepted, improved),
                flush=True,
            )

            if improved:
                best_state = pcopy_state(train_state)
                jax.block_until_ready(best_state)
                best_accepted = True
                best_triggered_loss = current_triggered_loss
                best_quiet_flow_ratio = quiet_flow_ratio
                best_quiet_action_drift = quiet_action_drift
                best_step = completed_step
                consecutive_non_improvements = 0
            else:
                consecutive_non_improvements += 1
                if not accepted:
                    rejected_validations += 1

            host_info = jax.device_get(info)
            logging.info(
                'step=%d train_loss=%.6f triggered=%.6f '
                'quiet_flow_ratio=%.4f quiet_drift=%.4f accepted=%s '
                'improved=%s patience=%d/%d',
                completed_step, float(np.asarray(host_info['loss'])),
                current_triggered_loss, quiet_flow_ratio, quiet_action_drift,
                accepted, improved, consecutive_non_improvements, patience,
            )

            if snapshot_callback is not None:
                snapshot_callback(
                    completed_step,
                    train_state,
                    {
                        'triggered_loss': current_triggered_loss,
                        'quiet_flow_ratio': quiet_flow_ratio,
                        'quiet_action_drift': quiet_action_drift,
                        'accepted': bool(accepted),
                        'improved': bool(improved),
                    },
                )

            if consecutive_non_improvements >= patience:
                stopped_early = True
                logging.info(
                    'early stop at step %d (%d non-improving validations)',
                    completed_step, consecutive_non_improvements,
                )
                break

        if not best_accepted:
            best_state = base_state

        metrics = {
            'accepted': best_accepted,
            'base_checkpoint': str(base_checkpoint),
            'base_triggered_loss': base_triggered_loss,
            'base_quiet_flow_loss': base_quiet_flow_loss,
            'best_triggered_loss': best_triggered_loss if best_accepted else None,
            'best_quiet_flow_ratio': best_quiet_flow_ratio if best_accepted else None,
            'best_quiet_action_drift': best_quiet_action_drift if best_accepted else None,
            'best_step': best_step,
            'validation_interval': validation_interval,
            'validations_run': validations_run,
            'rejected_validations': rejected_validations,
            'patience': patience,
            'stopped_early': stopped_early,
            'quiet_flow_loss_ratio_limit': quiet_flow_loss_ratio_limit,
            'quiet_action_drift_limit': quiet_action_drift_limit,
            'supervise_first_k': int(first_k),
        }
        return best_state, metrics
