# Derived from VLA-Arena's OpenPI training entry point (Apache-2.0).
"""Stage 4 entry point: one guarded LoRA update from a record set.

    failbank-train --records <record_set> --fold 0 \
        --base-checkpoint <ckpt>/pi05_vla_arena_finetuned --output-root <out> \
        [--steps 800 --batch-size 32 --quiet-weight 0.0 --data-seed 1]

``<record_set>`` is the output of ``failbank-build-round`` (or any root holding
``blobs/`` and ``derived/folds/offset_<fold>/{train,validation}.jsonl``).

Strict LoRA-only: every parameter except the LoRA matrices is frozen (vision tower, action
head, LLM base). Training starts fresh from ``--base-checkpoint``, which is never written.
The adapter is saved to ``<out>/offset_<fold>/params`` only if the held-out guard accepted
a validation point; otherwise ``metrics_rejected.json`` is written and the run exits
non-zero. Fold the adapter with ``failbank-fold`` before evaluation.

The paper's method: ``--steps 800 --batch-size 32 --quiet-weight 0.0
--validation-interval 800 --patience 99`` and data seeds 1, 2, 3 (three replicates).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from typing import Any

import flax.nnx as nnx
import jax
import orbax.checkpoint as ocp
import vla_arena.models.openpi.src.openpi.shared.nnx_utils as _nnx_utils
import vla_arena.models.openpi.src.openpi.training.config as _config
import vla_arena.models.openpi.src.openpi.training.optimizer as _optimizer
import vla_arena.models.openpi.src.openpi.training.weight_loaders as _weight_loaders

from failbank.train.derived_lora_data import (
    DerivedDataLoader,
    DerivedRecordDataset,
    create_derived_data_config,
)
from failbank.train.guard import train_with_heldout_guard

# Language instruction of the task the paper's bank was collected on
# (safety_static_obstacles, level 1, task 2). Records do not store it.
DEFAULT_PROMPT = "pick the mango on the table center and place it on the bowl"


def lora_config(lora_config_name: str, base_checkpoint: pathlib.Path):
    """The canonical LoRA recipe the Arena checkpoints were finetuned with, made strict.

    ``get_freeze_filter`` only freezes '.*llm.*', which would leave the ~410M-parameter
    SigLIP tower trainable. ``Not(.*lora.*)`` freezes everything but the ~50M LoRA
    parameters.
    """
    base = _config.get_config(lora_config_name)
    strict_lora_freeze = _nnx_utils.PathRegex(".*lora.*")
    return dataclasses.replace(
        base,
        name=f"{lora_config_name}_phase2_lora_dagger",
        freeze_filter=nnx.Not(strict_lora_freeze),
        ema_decay=None,
        batch_size=32,
        num_train_steps=400,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=20, peak_lr=3e-5, decay_steps=400, decay_lr=3e-6
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=_weight_loaders.CheckpointWeightLoader(
            str(base_checkpoint / "params")
        ),
    )


def _flat_state_entries(state):
    flat = state.flat_state()
    if hasattr(flat, "items"):
        return list(flat.items())
    return list(flat)


def assert_adapter_trainable(state, trainable_filter) -> None:
    """Every trainable parameter must be a LoRA parameter."""
    entries = _flat_state_entries(state.filter(trainable_filter))
    if not entries:
        raise ValueError("Trainable parameter set is empty")
    non_lora = ["/".join(str(p) for p in path) for path, _ in entries
                if "lora" not in "/".join(str(p) for p in path).lower()]
    if non_lora:
        raise ValueError("Non-LoRA parameters are trainable (strict LoRA-only violated): "
                         + ", ".join(non_lora[:20]))


def assert_paths(*, base_checkpoint: pathlib.Path, output_path: pathlib.Path) -> None:
    base = base_checkpoint.resolve()
    output = output_path.resolve()
    if output == base:
        raise ValueError("Adapter output path equals the base checkpoint path")
    if output.is_relative_to(base):
        raise ValueError("Adapter output path is inside the base checkpoint")
    if base.is_relative_to(output):
        raise ValueError("Base checkpoint is inside the adapter output path")


def save_adapter_only(state, config, output_path: pathlib.Path, base_checkpoint: pathlib.Path) -> None:
    assert_paths(base_checkpoint=base_checkpoint, output_path=output_path)
    adapter_state = state.params.filter(config.trainable_filter)
    if not _flat_state_entries(adapter_state):
        raise ValueError("Refusing to save an empty adapter")
    output_path.mkdir(parents=True, exist_ok=False)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(str(output_path / "params"), adapter_state.to_pure_dict())
    checkpointer.wait_until_finished()


def _resolve_fold_manifest(root: pathlib.Path, offset: int, split: str) -> pathlib.Path:
    names = (split, "val") if split == "validation" else (split,)
    candidates = []
    for name in names:
        candidates += [root / f"offset_{offset}" / f"{name}.jsonl",
                       root / f"offset_{offset:02d}" / f"{name}.jsonl",
                       root / f"{name}_offset_{offset}.jsonl"]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"no {split!r} manifest for fold {offset}; checked:\n"
                            + "\n".join(f"  - {c}" for c in candidates))


def _resolve_asset_id(assets_dir: pathlib.Path, asset_id: str | None) -> str:
    """Pick the norm-stats directory; the Arena pi0.5 and pi0 checkpoints lay it out differently."""
    if asset_id:
        return asset_id
    found = sorted(p.parent.relative_to(assets_dir).as_posix()
                   for p in assets_dir.rglob("norm_stats.json"))
    if len(found) != 1:
        raise ValueError(f"cannot infer --asset-id under {assets_dir}: found {found}")
    return found[0]


def _make_loader_factory(*, manifest_path, records_root, data_config, global_batch_size,
                         quiet_weight, verify_sha256, primary_image_key, wrist_image_key,
                         seed, num_workers, prompt, action_horizon):
    process_count = jax.process_count()
    if global_batch_size % process_count != 0:
        raise ValueError(f"Global batch size {global_batch_size} not divisible by "
                         f"process count {process_count}")
    local_batch_size = global_batch_size // process_count

    def factory(data_sharding):
        dataset = DerivedRecordDataset(
            manifest_path, records_root=records_root, data_config=data_config,
            quiet_weight=quiet_weight, verify_sha256=verify_sha256, prompt=prompt,
            action_horizon=action_horizon)
        return DerivedDataLoader(
            dataset, batch_size=local_batch_size, sharding=data_sharding, seed=seed,
            num_workers=num_workers, drop_last=False,
            primary_image_key=primary_image_key, wrist_image_key=wrist_image_key)

    return factory


def train_fold(args, offset: int) -> dict[str, Any]:
    base_checkpoint = args.base_checkpoint
    config = lora_config(args.lora_config, base_checkpoint)
    # The run name is only a label; kept identical to the original trainer.
    config = dataclasses.replace(config, name=f"pi05_vla_arena_phase2_lora_offset_{offset}")
    if args.steps > 0:
        config = dataclasses.replace(config, num_train_steps=args.steps)
    if args.batch_size > 0:
        config = dataclasses.replace(config, batch_size=args.batch_size)
    state_shape = jax.eval_shape(lambda rng: nnx.state(config.model.create(rng)), jax.random.key(123))
    assert_adapter_trainable(state_shape, config.trainable_filter)

    train_manifest = _resolve_fold_manifest(args.derived_root, offset, "train")
    validation_manifest = _resolve_fold_manifest(args.derived_root, offset, "validation")
    asset_id = _resolve_asset_id(args.assets_dir, args.asset_id)
    data_config = create_derived_data_config(config, assets_dirs=args.assets_dir, asset_id=asset_id)

    # Data order derives from the fold index unless --data-seed is given (replicates).
    seed_base = offset if args.data_seed is None else args.data_seed
    common = dict(records_root=args.blobs_root, data_config=data_config,
                  global_batch_size=config.batch_size, quiet_weight=args.quiet_weight,
                  verify_sha256=args.verify_sha256, primary_image_key=args.primary_image_key,
                  wrist_image_key=args.wrist_image_key, num_workers=args.num_workers,
                  prompt=args.prompt, action_horizon=config.model.action_horizon)
    train_factory = _make_loader_factory(manifest_path=train_manifest, seed=seed_base + 1000, **common)
    validation_factory = _make_loader_factory(manifest_path=validation_manifest, seed=seed_base, **common)

    state, metrics = train_with_heldout_guard(
        config,
        base_checkpoint=base_checkpoint,
        train_loader_factory=train_factory,
        validation_loader_factory=validation_factory,
        validation_interval=args.validation_interval or config.num_train_steps,
        patience=args.patience,
        quiet_flow_loss_ratio_limit=args.quiet_flow_loss_ratio_limit,
        quiet_action_drift_limit=args.quiet_action_drift_limit,
    )
    metrics.update({
        "offset": offset,
        "train_manifest": str(train_manifest),
        "validation_manifest": str(validation_manifest),
        "records_root": str(args.blobs_root),
        "assets_dir": str(args.assets_dir),
        "asset_id": asset_id,
        "action_space": "environment",
        "quiet_weight": args.quiet_weight,
        "data_seed": seed_base,
        "base_model": args.lora_config,
        "update_method": "gemma_300m_lora_dagger",
        "num_train_steps": int(config.num_train_steps),
        "batch_size": int(config.batch_size),
    })

    output_path = args.output_root / f"offset_{offset}"
    if not metrics["accepted"]:
        output_path.mkdir(parents=True, exist_ok=True)
        rejected = output_path / "metrics_rejected.json"
        rejected.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str))
        print(f"GUARD_REJECT offset={offset} limits(drift={args.quiet_action_drift_limit} "
              f"flow={args.quiet_flow_loss_ratio_limit}) metrics={rejected}", flush=True)
        raise SystemExit(f"fold {offset}: no validation point passed the held-out guard; "
                         "no adapter saved")

    assert_adapter_trainable(state.params, config.trainable_filter)
    save_adapter_only(state, config, output_path, base_checkpoint)
    with (output_path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"ADAPTER_SAVED {output_path} best_step={metrics['best_step']} "
          f"flow_ratio={metrics['best_quiet_flow_ratio']:.5f} "
          f"drift={metrics['best_quiet_action_drift']:.6f}", flush=True)
    return metrics


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--records", type=pathlib.Path,
                   help="record-set root (implies --blobs-root <r>/blobs --derived-root <r>/derived/folds)")
    p.add_argument("--blobs-root", type=pathlib.Path)
    p.add_argument("--derived-root", type=pathlib.Path)
    p.add_argument("--fold", "--offset", dest="fold", type=int, default=0,
                   help="held-out offset whose fold to train on (paper: 0)")
    p.add_argument("--base-checkpoint", type=pathlib.Path, required=True,
                   help="starting checkpoint (params/ + assets/); never written")
    p.add_argument("--lora-config", default="pi05_vla_arena_low_mem_finetune",
                   help="OpenPI LoRA config; pi0 uses pi0_vla_arena_low_mem_finetune")
    p.add_argument("--output-root", type=pathlib.Path, required=True)
    p.add_argument("--assets-dir", type=pathlib.Path, default=None,
                   help="default: <base-checkpoint>/assets")
    p.add_argument("--asset-id", default=None, help="default: inferred from assets-dir")
    p.add_argument("--primary-image-key", default="base_0_rgb")
    p.add_argument("--wrist-image-key", default="left_wrist_0_rgb")
    p.add_argument("--steps", "--smoke-steps", dest="steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--quiet-weight", type=float, default=0.0,
                   help="weight of untriggered (quiet) records; lambda_q in the paper")
    p.add_argument("--data-seed", type=int, default=None)
    p.add_argument("--validation-interval", type=int, default=0,
                   help="0 = validate once, at the last step (the paper's setting)")
    p.add_argument("--patience", type=int, default=99)
    p.add_argument("--quiet-flow-loss-ratio-limit", type=float, default=1.10)
    p.add_argument("--quiet-action-drift-limit", type=float, default=0.05)
    p.add_argument("--verify-sha256", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="task instruction injected into records (they do not store it)")
    args = p.parse_args(argv)
    if args.records is not None:
        args.blobs_root = args.blobs_root or args.records / "blobs"
        args.derived_root = args.derived_root or args.records / "derived" / "folds"
    if args.blobs_root is None or args.derived_root is None:
        p.error("give --records, or both --blobs-root and --derived-root")
    return args


def main(argv=None) -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)
    args.base_checkpoint = args.base_checkpoint.resolve()
    args.blobs_root = args.blobs_root.resolve()
    args.derived_root = args.derived_root.resolve()
    args.output_root = args.output_root.resolve()
    args.assets_dir = (args.assets_dir or args.base_checkpoint / "assets").resolve()
    fold_dir = args.derived_root / f"offset_{args.fold}"
    if not (fold_dir / "train.jsonl").exists():
        raise SystemExit(f"no fold {fold_dir}")
    if (fold_dir / "validation.jsonl").stat().st_size == 0:
        raise SystemExit(f"fold {fold_dir} has an empty validation split")
    if args.quiet_weight < 0:
        raise SystemExit("--quiet-weight must be non-negative")
    assert_paths(base_checkpoint=args.base_checkpoint, output_path=args.output_root)
    train_fold(args, args.fold)


if __name__ == "__main__":
    main()
