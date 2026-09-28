#!/usr/bin/env python3
"""Stage 1 and evaluation: run one VLA-Arena cell with an observe-only teacher.

    failbank-rollout --output <cell>/result.json --task-suite safety_static_obstacles \
        --task-level 1 --task-id 2 --offset 0 [--policy-checkpoint <folded ckpt>] \
        [--record-root <collection>/records] [--sampler-advance 1]

One process runs one cell (one task, one initial-state offset, ``--trials`` episodes) of
an OpenPI policy (pi0.5 by default, pi0 with ``--openpi-config pi0``) through VLA-Arena's
own evaluator. The only thing inserted is the evaluator's step seam
(``runtime_env_step``, added by ``patches/vla-arena-2ddcb00.patch``). At every control step:

    features  = read-only cost-relation features of the current state
    proposal  = teacher.propose(state, a_t)          -> (~a_t, z_t)       [--teacher]
    executed  = a_t  (observe-only, default)  or ~a_t (--execute teacher)
    env.step(executed)

With ``--record-root`` every step is also written as a Stage-1 learning record (model
inputs, the policy's action chunk, a_t, ~a_t, the projection, progress and cost features);
the episode is committed atomically once ``result.json`` has been written with status
"pass".

Outputs next to ``result.json``:

    result.json        SR/official cost/policy-induced cost of the cell, identity and
                       provenance gates, policy-server evidence (fails closed: a crashed
                       cell writes status "fail" and no scientific fields)
    steps.jsonl        one row per control step (feeds the policy-induced cost)
    teacher.jsonl      the teacher's per-step verdicts and a per-episode trigger summary
    server_port*.log   the policy server's log (the restored checkpoint path is in it)

Reproducibility: ``--seed`` does not change the simulation (Arena initial states are
fixed per offset). With the deterministic server flags installed here a cell is
bit-reproducible on one GPU model; different GPU models give different trajectories.
Pin every cell of a comparison to one host. Independent repeats of a cell come from
``--sampler-advance N``, which draws N extra policy samples before the episode starts.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import os
import sys
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import failbank
from failbank.runtime.arena_hooks import cost_relation_features, to_list
from failbank.teacher.base import TeacherContext, default_candidates, load_teacher

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
OPENPI_CONFIGS = {"pi05": CONFIG_DIR / "openpi_pi05.yaml", "pi0": CONFIG_DIR / "openpi_pi0.yaml"}
SUPPORTED_TRAIN_CONFIGS = ("pi05_vla_arena", "pi0_vla_arena")
RAW_OBS_KEYS = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
                "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos", "robot0_joint_vel")
MODEL_INPUT_KEYS = ("observation/image", "observation/wrist_image", "observation/state")


# --------------------------------------------------------------------------- policy clients

class FirstActionClient:
    """Expose one unchanged canonical action per inference (replan every step)."""

    def __init__(self, client, adapter):
        self.client = client
        self.adapter = adapter
        self.telemetry: list[dict] = []
        self.errors: list[str] = []

    def infer(self, element):
        from failbank.runtime.action_adapter import adapt_first_action
        try:
            result = self.client.infer(element)
            action, telemetry = adapt_first_action(self.adapter, result["actions"])
            self.telemetry.append(telemetry)
            return {**result, "actions": np.asarray([action])}
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            raise


class NativeChunkClient:
    """Validate and forward the unmodified native chunk (replan_steps=5 reference)."""

    def __init__(self, client):
        self.client = client
        self.telemetry: list[dict] = []
        self.errors: list[str] = []

    def infer(self, element):
        try:
            result = self.client.infer(element)
            actions = np.asarray(result["actions"])
            assert actions.ndim == 2 and actions.shape[1] == 7 and len(actions) >= 5
            self.telemetry.append({"chunk_length": int(len(actions))})
            return {**result, "actions": actions.copy()}
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            raise


def telemetry_summary(rows: list[dict]) -> dict:
    return {
        "inference_chunks": len(rows),
        "first_actions_executed": len(rows),
        "replans_requested": sum(bool(row["replan"]) for row in rows),
        "discarded_actions_total": sum(row["discarded_actions"] for row in rows),
        "chunk_lengths": sorted(set(row["chunk_length"] for row in rows)),
        "controller_effective_translation_abs_max_m": max(
            (row["controller_effective_translation_abs_max_m"] for row in rows), default=None),
        "controller_effective_rotation_abs_max_rad": max(
            (row["controller_effective_rotation_abs_max_rad"] for row in rows), default=None),
        "controller_saturation_total": sum(row["controller_saturation_count"] for row in rows),
        "all_actions_forwarded_unchanged": bool(rows)
        and all(row["action_forwarded_unchanged"] for row in rows),
        "corrections_applied": sum(bool(row["correction_applied"]) for row in rows),
        "physical_units_gate": "pass"
        if rows and all(row["physical_units_gate"] == "pass" for row in rows) else "fail",
        "samples": rows[:3] + rows[-3:] if len(rows) > 3 else rows,
    }


class InferStash:
    """Remember the raw websocket client and the last observation it was asked about.

    Only installed for ``--sampler-advance``: the advance re-asks the identical question N
    times before the episode's first step so that the server's sampler is in a different
    state, giving an independent repeat of the same policy on the same cell.
    """

    def __init__(self):
        self.client = None
        self.element = None

    def install(self):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        original = WebsocketClientPolicy.infer
        stash = self

        def infer(client_self, element, *a, **kw):
            stash.client = client_self
            stash.element = element
            return original(client_self, element, *a, **kw)

        WebsocketClientPolicy.infer = infer


# --------------------------------------------------------------------------- telemetry

class JsonlWriter:
    def __init__(self, path: Path | None, truncate: bool = True):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if truncate and path.exists():
                path.unlink()

    def write(self, row: dict) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- Stage-1 records

class RecordCapture:
    """Writes one Stage-1 learning record per control step (fail-open for the rollout).

    Called strictly after the environment step. Any exception disables the episode's
    record shard (it will never get a COMPLETE marker) but never changes the rollout.
    """

    def __init__(self, root: Path, *, label: str, seed: int, task_suite: str, observe_only: bool):
        from failbank.records.input_capture import InferCapture
        self.root = Path(root)
        self.label = label
        self.seed = seed
        self.task_suite = task_suite
        self.observe_only = observe_only
        self.infer = InferCapture()
        self.recorder = None
        self.progress = None
        self.progress_pre = None
        self.step_index = 0
        self.init_done = False
        self.preflight_done = False
        self.error: str | None = None

    def install(self) -> None:
        """Must run before the first policy query: infer #k is popped at step k."""
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        self.infer.wrap_class(WebsocketClientPolicy)

    def _cost_pair_snapshot(self, env, obs) -> dict:
        # Read AFTER env.step: the distance is that of the post-step state (this is what
        # the paper's records carry; `first_risk_step` stages on it).
        try:
            f = cost_relation_features(env, obs)
            return {
                "cost_pair_min_distance": f.get("cost_pair_min_distance"),
                "cost_pair_min_name": f.get("cost_pair_min_name"),
                "cost_pair_contacts": f.get("cost_pair_contacts"),
                "cost_pair_count": f.get("cost_pair_count"),
                # so a later reader cannot mistake `executed_action` for what actually ran
                "shield_observe_only": self.observe_only,
            }
        except Exception as exc:
            sys.stderr.write(f"cost-pair snapshot failed: {exc}\n")
            return {}

    def capture(self, env, obs, next_obs, done, info, t, nominal, proposal, teacher,
                task_description: str) -> None:
        from failbank.records.goal_progress import extract_goal, signed_distance_to_region
        from failbank.records.progress_capture import ProgressCapture
        from failbank.records.recorder import EpisodeRecorder
        from failbank.records.schema_v1 import SCHEMA_VERSION, EpisodeRaw, RawStep
        try:
            if not self.init_done:
                self.init_done = True
                self.step_index = 0
                now = datetime.now(timezone.utc).isoformat()
                eid = hashlib.sha256(f"{self.label}|{os.getpid()}|{now}".encode()).hexdigest()[:24]
                prog = ProgressCapture(extract_goal=extract_goal,
                                       signed_distance_to_region=signed_distance_to_region)
                goal = prog.reset(env)
                self.progress = prog
                self.progress_pre = prog.read(obs)
                ep = EpisodeRaw(
                    schema_version=SCHEMA_VERSION, episode_id=eid,
                    task_description=str(task_description), seed=int(self.seed),
                    runtime_method="formal_late_margin",
                    config={"candidate_set": ["nominal", "late_margin", "late_min", "late_soft", "eef_wide"],
                            "oracle_radius": 0.04},
                    goal=goal.as_json(), started_at_utc=now,
                    metadata={"run_label": self.label, "task_suite": self.task_suite,
                              "teacher": getattr(teacher, "name", None),
                              "observe_only": self.observe_only,
                              "failbank_version": failbank.__version__})
                self.recorder = EpisodeRecorder(dataset_root=self.root, episode=ep, enabled=True)
            idx = self.step_index
            self.step_index = idx + 1
            rec = self.recorder
            if rec is None or not rec.enabled:
                return
            if proposal is None or not proposal.projection:
                raise RuntimeError("formal projection unavailable; fail-closed")
            executed = list(proposal.action)
            triggered = bool(proposal.triggered)
            step_cost = float((info or {}).get("cost", 0.0) or 0.0)
            prog = self.progress
            progress_post = prog.read(next_obs)
            progress_pre = self.progress_pre if self.progress_pre is not None else prog.read(obs)
            captured = self.infer.pop(idx)
            if not self.preflight_done:
                if captured is None or captured.action_chunk.shape[0] < 1 or not np.array_equal(
                        np.asarray(captured.action_chunk[0]), np.asarray(nominal)):
                    raise RuntimeError("record preflight failed: infer/chunk misalignment at step 0")
                self.preflight_done = True
            if hasattr(teacher, "candidates"):
                candidates = teacher.candidates(nominal, proposal)
            else:
                candidates = default_candidates(nominal, proposal)
            raw_refs = {k: rec.dump_array(np.asarray(obs[k])) for k in RAW_OBS_KEYS if k in obs}
            model_refs, chunk_ref, prompt = {}, None, ""
            if captured is not None:
                for k in MODEL_INPUT_KEYS:
                    if k in captured.element:
                        model_refs[k] = rec.dump_array(np.asarray(captured.element[k]))
                chunk_ref = rec.dump_array(np.asarray(captured.action_chunk))
                prompt = str(captured.element.get("prompt", ""))
            goal = prog.goal
            objpos = {k: np.asarray(obs[k], dtype=float).tolist()
                      for k in (f"{goal.object_name}_pos", "new_bowl_1_pos", "white_yellow_mug_1_pos")
                      if k in obs}
            formal = dict(proposal.projection)
            formal["triggered"] = triggered
            step = RawStep(
                schema_version=SCHEMA_VERSION, episode_id=rec.episode.episode_id,
                step_index=int(t), infer_call_index=idx, task_description=prompt,
                raw_observation_refs=raw_refs, model_input_refs=model_refs,
                nominal_action_chunk_ref=chunk_ref,
                nominal_action=[float(x) for x in nominal],
                executed_action=[float(x) for x in executed],
                formal_projection=formal, projection_candidates=candidates,
                progress_pre=float(progress_pre), progress_post=float(progress_post),
                progress_delta=float(progress_post - progress_pre), goal=goal.as_json(),
                object_positions=objpos, reward=0.0, done=bool(done),
                runtime_info={"cost": step_cost, **self._cost_pair_snapshot(env, obs)},
                recorder_diagnostics={})
            rec.append(step)
            self.progress_pre = progress_post
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            if self.recorder is not None:
                try:
                    self.recorder._disable(exc)
                except Exception:
                    pass
            else:
                sys.stderr.write(f"record capture error: {exc}\n")

    def status(self) -> dict:
        rec = self.recorder
        return {"root": str(self.root), "committed": False, "error": self.error,
                "episode_id": None if rec is None else rec.episode.episode_id,
                "steps_written": 0 if rec is None else rec.step_count,
                "enabled": bool(rec is not None and rec.enabled)}

    def finalize(self, result_path: Path, status: str) -> dict:
        rec = self.recorder
        out = self.status()
        if rec is None:
            out["error"] = out["error"] or "no step was captured"
            return out
        if status != "pass":
            # A crashed cell is not an outcome: leave the shard uncommitted (no COMPLETE),
            # so the record builder skips it instead of labelling it a failure.
            out["error"] = out["error"] or "cell status != pass; episode not committed"
            return out
        out["committed"] = bool(rec.finalize(result_path=result_path))
        if not out["committed"]:
            out["error"] = rec.disabled_reason or out["error"]
        return out


# --------------------------------------------------------------------------- the step seam

class StepHook:
    """Replacement for the evaluator's ``runtime_env_step(env, obs, task, action, cfg, t)``."""

    def __init__(self, *, teacher, execute: str, episode_label: str, task_suite: str,
                 steps: JsonlWriter, teacher_log: JsonlWriter, records: RecordCapture | None,
                 sampler_advance: int, stash: InferStash | None, out_dir: Path = Path(".")):
        self.teacher = teacher
        self.out_dir = out_dir
        self.execute = execute
        self.episode_label = episode_label
        self.task_suite = task_suite
        self.steps = steps
        self.teacher_log = teacher_log
        self.records = records
        self.sampler_advance = int(sampler_advance)
        self.stash = stash
        self._advanced = False
        self._last_t: int | None = None
        self._episode = 0
        self._stats = {"steps": 0, "triggers": 0, "norm_sum": 0.0}

    def _flush_summary(self) -> None:
        if self.teacher is None or not self._stats["steps"]:
            return
        n = self._stats["steps"]
        self.teacher_log.write({
            "type": "teacher_summary", "episode_id": self.episode_label, "episode": self._episode,
            "teacher": getattr(self.teacher, "name", None), "steps": n,
            "triggers": self._stats["triggers"], "trigger_rate": self._stats["triggers"] / n,
            "norm_mean": self._stats["norm_sum"] / n})
        self._stats = {"steps": 0, "triggers": 0, "norm_sum": 0.0}

    def close(self) -> None:
        self._flush_summary()
        audit = getattr(self.teacher, "audit", None)
        if callable(audit):
            self.teacher_log.write({"type": "teacher_audit", "episode_id": self.episode_label,
                                    **audit()})

    def __call__(self, env, obs, task_description, action, cfg, t):
        t = int(t)
        features = cost_relation_features(env, obs)
        nominal = to_list(action)
        ctx = TeacherContext(env=env, obs=obs, features=features, step=t,
                             task_description=str(task_description), task_suite=self.task_suite,
                             out_dir=str(self.out_dir), episode_label=self.episode_label)
        if self._last_t is None or t <= self._last_t:
            if self._last_t is not None:
                self._flush_summary()
                self._episode += 1
            if self.teacher is not None:
                self.teacher.reset(ctx)
        self._last_t = t

        if self.sampler_advance and not self._advanced:
            self._advanced = True
            if self.stash is None or self.stash.client is None:
                raise RuntimeError("--sampler-advance needs the infer stash; it never saw a query")
            for _ in range(self.sampler_advance):
                self.stash.client.infer(self.stash.element)
            sys.stderr.write("SAMPLER_ADVANCED n=%d\n" % self.sampler_advance)

        proposal = self.teacher.propose(ctx, nominal) if self.teacher is not None else None
        to_env = nominal if (self.execute == "nominal" or proposal is None) else list(proposal.action)
        next_obs, reward, done, info = env.step(to_env)
        info = dict(info or {})
        step_cost = float(info.get("cost", 0.0) or 0.0)

        self.steps.write({
            "episode_id": self.episode_label, "step_idx": t, "done": bool(done),
            "step_cost": step_cost,
            "pre_step_features": {"arena_timestep": t, **features},
            "nominal_action": nominal, "executed_action": list(to_env),
            "teacher_triggered": None if proposal is None else bool(proposal.triggered),
        })
        if proposal is not None:
            p = proposal.projection
            self._stats["steps"] += 1
            self._stats["triggers"] += int(proposal.triggered)
            self._stats["norm_sum"] += float(proposal.norm)
            self.teacher_log.write({
                "type": "teacher_step", "episode_id": self.episode_label, "step_idx": t,
                "triggered": bool(proposal.triggered), "norm": float(proposal.norm),
                "min_barrier": p.get("min_barrier"), "active_constraints": p.get("active_constraints"),
                "status": p.get("status"), **({"metadata": proposal.metadata} if proposal.metadata else {})})
        if self.records is not None:
            self.records.capture(env, obs, next_obs, done, info, t, nominal, proposal,
                                 self.teacher, task_description)
        return next_obs, float(reward), bool(done), info


# --------------------------------------------------------------------------- one cell

def load_openpi_config(spec: str) -> tuple[Path, dict]:
    import yaml
    path = OPENPI_CONFIGS.get(spec, Path(spec))
    raw = yaml.safe_load(Path(path).read_text())
    raw["policy_checkpoint_dir"] = os.path.expandvars(str(raw["policy_checkpoint_dir"]))
    if "$" in raw["policy_checkpoint_dir"]:
        raise SystemExit(f"{path}: policy_checkpoint_dir {raw['policy_checkpoint_dir']!r} has an "
                         "unset variable; export FAILBANK_CHECKPOINTS (see README)")
    return Path(path), raw


def run_cell(args, evaluator, hook: StepHook, report: dict) -> None:
    from vla_arena.models.openpi.src.openpi.training import config as openpi_config

    from failbank.runtime.action_adapter import ArenaControllerActionContract, Pi05CanonicalActionAdapter
    from failbank.runtime.contract import assert_architecture, assert_report_identity, load_task_and_states
    from failbank.runtime.metrics import decompose_episode
    from failbank.runtime.server import assert_episode_transport_clean, connect_policy_server

    config_path, raw = load_openpi_config(args.openpi_config)
    assert raw["train_config_name"] in SUPPORTED_TRAIN_CONFIGS, (
        f"unsupported train_config_name={raw['train_config_name']!r}; expected one of "
        f"{SUPPORTED_TRAIN_CONFIGS}")
    base_checkpoint = Path(raw["policy_checkpoint_dir"])
    checkpoint = Path(args.policy_checkpoint) if args.policy_checkpoint else base_checkpoint
    cfg_obj = openpi_config.get_config(raw["train_config_name"])
    # Architecture is a property of the BASE checkpoint; an adapter/folded checkpoint is
    # checked for metadata only.
    report["architecture"] = assert_architecture(cfg_obj, base_checkpoint / "params/_METADATA")
    report["openpi_config"] = str(config_path)
    if checkpoint != base_checkpoint:
        assert (checkpoint / "params/_METADATA").exists(), f"no params/_METADATA under {checkpoint}"
        report["architecture"]["base_checkpoint_dir"] = str(base_checkpoint)
        report["architecture"]["adapter_checkpoint_dir"] = str(checkpoint)
    cfg = replace(
        evaluator.GenerateConfig(**raw),
        task_suite_name=args.task_suite, task_level=args.task_level, num_trials_per_task=1,
        init_state_selection_mode="episode_idx", init_state_offset=args.offset,
        init_state_offset_random=False, replan_steps=args.replan_steps, save_video_mode="none",
        use_local_log=False, seed=args.seed, policy_checkpoint_dir=str(checkpoint),
    )
    suite = evaluator.benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task, initial_states, identity = load_task_and_states(suite, cfg, args.task_id)
    report["task_identity"] = identity
    assert_report_identity(report, cfg, args.task_id)

    client, source, selected_config, report["_managed_process"], cfg, server_evidence = connect_policy_server(
        cfg, base_port=args.port_base, array_task_id=args.array_task_id, attempts=args.server_attempts,
        is_port_open=evaluator._is_port_open, is_ready=evaluator._is_websocket_ready,
        create_client=evaluator._create_policy_client,
        stop_process=lambda process: evaluator._stop_managed_policy_server(process, timeout_sec=3.0),
    )
    assert server_evidence["readiness_confirmed"] is True
    report.update({"policy_server_ready": True, "policy_server": server_evidence,
                   "policy_source": source, "policy_port": cfg.port,
                   "selected_config": selected_config,
                   "loader_full_tree_shape_validation": "pass"})
    env, _ = evaluator.get_vla_arena_env(task, resolution=evaluator.VLA_ARENA_ENV_RESOLUTION)
    report["_env"] = env
    task_description = task.language[0] if isinstance(task.language, list) else task.language
    contract = ArenaControllerActionContract.from_env(env)
    report["controller_contract"] = contract.to_dict()
    wrapped = (FirstActionClient(client, Pi05CanonicalActionAdapter(contract))
               if args.replan_steps == 1 else NativeChunkClient(client))
    episode_reports, combined_log = [], []
    for episode_idx in range(args.trials):
        initial_state_idx = evaluator.select_init_state_index(
            num_initial_states=len(initial_states), episode_idx=episode_idx,
            selection_mode=cfg.init_state_selection_mode, offset=cfg.init_state_offset,
            offset_random=False, rng=np.random.default_rng(cfg.seed))
        log = io.StringIO()
        success, _, cost = evaluator.run_episode(
            cfg, env, task_description, {}, initial_states[initial_state_idx], log, wrapped)
        text = log.getvalue()
        combined_log.extend(text.splitlines())
        assert_episode_transport_clean(text, wrapped.errors)
        episode_reports.append({"episode": episode_idx + 1, "initial_state_index": initial_state_idx,
                                "success": bool(success), "official_cost": float(cost)})
    hook.close()
    text = "\n".join(combined_log)
    if args.replan_steps == 1:
        adapter_report = telemetry_summary(wrapped.telemetry)
    else:
        adapter_report = {"inference_chunks": len(wrapped.telemetry),
                          "chunk_lengths": sorted({x["chunk_length"] for x in wrapped.telemetry}),
                          "all_actions_forwarded_unchanged": True, "corrections_applied": 0,
                          "physical_units_gate": "pass", "native_replan_steps": 5}
    report.update({
        "adapter": adapter_report, "inference_errors": wrapped.errors,
        "successes": sum(int(row["success"]) for row in episode_reports),
        "cost": sum(row["official_cost"] for row in episode_reports),
        "episodes": episode_reports,
        "episode_finished_marker": "Episode finished after" in text,
        "episode_error_marker": "Episode error:" in text,
        "log_tail": text.splitlines()[-30:],
    })
    assert_report_identity(report, cfg, args.task_id)
    assert report["episode_finished_marker"] and not report["episode_error_marker"]
    assert not wrapped.errors and adapter_report["inference_chunks"] > 0
    if args.replan_steps == 1:
        assert adapter_report["first_actions_executed"] == adapter_report["replans_requested"]
    # The POLICY's action reaches the seam unmodified; only the seam decides what executes.
    assert adapter_report["all_actions_forwarded_unchanged"]
    assert adapter_report["corrections_applied"] == 0
    assert adapter_report["physical_units_gate"] == "pass"

    rows = [json.loads(line) for line in hook.steps.path.read_text().splitlines() if line.strip()]
    split, current, previous = [], [], None
    for row in rows:
        step = int(row["step_idx"])
        if current and step <= previous:
            split.append(current)
            current = []
        current.append(row)
        previous = step
    if current:
        split.append(current)
    assert len(split) == args.trials, f"telemetry episode count mismatch: {len(split)}"
    for episode, episode_rows in zip(episode_reports, split):
        metric = decompose_episode(episode_rows)
        episode["policy_induced_cost"] = metric["policy_induced_cost"]
        episode["initial_attributed_cost"] = metric["initial_attributed_cost"]
    report["metric_decomposition"] = {
        "official_sr": report["successes"] / args.trials,
        "official_cc": report["cost"] / args.trials,
        "policy_induced_cc": sum(row["policy_induced_cost"] for row in episode_reports) / args.trials,
    }
    report["episodes_completed"] = args.trials
    report["status"] = "pass"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--output", type=Path, required=True, help="result.json path")
    p.add_argument("--task-suite", default="safety_static_obstacles")
    p.add_argument("--task-level", type=int, default=1)
    p.add_argument("--task-id", type=int, default=2)
    p.add_argument("--offset", type=int, required=True, help="initial-state offset")
    p.add_argument("--seed", type=int, default=17, help="recorded only; does not change the simulation")
    p.add_argument("--replan-steps", type=int, choices=(1, 5), default=1)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--openpi-config", default="pi05",
                   help="pi05 | pi0 | path to an evaluation yaml")
    p.add_argument("--policy-checkpoint", type=Path, default=None,
                   help="checkpoint to evaluate (e.g. a folded adapter); default: the yaml's base")
    p.add_argument("--port-base", type=int, default=18000)
    p.add_argument("--array-task-id", type=int, default=1)
    p.add_argument("--server-attempts", type=int, default=3)
    p.add_argument("--server-timeout", type=float, default=1800.0,
                   help="floor for the policy-server readiness wait, seconds")
    p.add_argument("--server-log-dir", type=Path, default=None, help="default: the output directory")
    t = p.add_argument_group("teacher")
    t.add_argument("--teacher", default="oracle_geometry",
                   help="oracle_geometry (paper) | none | package.module:Class")
    t.add_argument("--execute", choices=("nominal", "teacher"), default="nominal",
                   help="nominal = observe-only (FailBank collection and evaluation); "
                        "teacher = the teacher's action is executed (in-loop shield baseline)")
    t.add_argument("--teacher-alpha", type=float, default=3.0)
    t.add_argument("--teacher-eef-radius", type=float, default=0.03)
    t.add_argument("--teacher-oracle-radius", type=float, default=0.04)
    t.add_argument("--teacher-margin", type=float, default=0.02)
    t.add_argument("--teacher-max-translation", type=float, default=1.0)
    t.add_argument("--teacher-override-eps", type=float, default=1e-6)
    r = p.add_argument_group("records / repeats")
    r.add_argument("--record-root", type=Path, default=None,
                   help="write Stage-1 learning records under this root")
    r.add_argument("--sampler-advance", type=int, default=0,
                   help="draw N extra policy samples before the episode (independent repeat)")
    r.add_argument("--episode-id", default=None, help="run label (default derived from the cell)")
    args = p.parse_args(argv)
    if args.record_root is not None and args.teacher == "none":
        p.error("--record-root needs a teacher: records carry the teacher's proposal")
    if args.record_root is not None and args.sampler_advance:
        p.error("--record-root and --sampler-advance cannot be combined: the extra queries "
                "desynchronise the per-step input capture")
    if args.record_root is not None and args.trials != 1:
        p.error("--record-root records one episode per process; use --trials 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_dir = args.output.parent
    label = args.episode_id or (f"{args.task_suite}_L{args.task_level}t{args.task_id}"
                                f"_off{args.offset}_adv{args.sampler_advance}")
    report: dict[str, Any] = {
        "status": "fail", "task_suite_name": args.task_suite, "task_level": args.task_level,
        "task_id": args.task_id, "init_state_offset": args.offset,
        "replan_steps": args.replan_steps, "episodes_requested": args.trials,
    }

    teacher = load_teacher(args.teacher, alpha=args.teacher_alpha,
                           eef_radius=args.teacher_eef_radius,
                           oracle_radius=args.teacher_oracle_radius,
                           obstacle_margin=args.teacher_margin,
                           max_translation=args.teacher_max_translation,
                           override_eps=args.teacher_override_eps)
    if getattr(teacher, "needs_aegis_cameras", False):
        # read by the patched get_vla_arena_env: adds the backview camera and depth planes
        os.environ["FAILBANK_AEGIS_CAMERAS"] = "1"
    report["failbank"] = {
        "version": failbank.__version__, "teacher": getattr(teacher, "name", None),
        "teacher_spec": args.teacher, "execute": args.execute,
        "teacher_config": (dataclasses.asdict(teacher.cfg)
                           if dataclasses.is_dataclass(getattr(teacher, "cfg", None)) else None),
        "sampler_advance": args.sampler_advance, "episode_label": label,
        "record_root": None if args.record_root is None else str(args.record_root),
        "policy_checkpoint": None if args.policy_checkpoint is None else str(args.policy_checkpoint),
    }

    from vla_arena.models.openpi import evaluator
    from failbank.runtime.server import fail_cell_atomically, install_evaluator_patches

    if not hasattr(evaluator, "runtime_env_step") or not hasattr(evaluator, "_is_websocket_ready"):
        raise SystemExit("VLA-Arena is not patched: apply patches/vla-arena-2ddcb00.patch (see README)")
    install_evaluator_patches(evaluator, timeout_floor_sec=args.server_timeout,
                              log_dir=args.server_log_dir or out_dir)
    stash = None
    if args.sampler_advance:
        stash = InferStash()
        stash.install()
    records = None
    if args.record_root is not None:
        records = RecordCapture(args.record_root, label=label, seed=args.seed,
                                task_suite=args.task_suite, observe_only=args.execute == "nominal")
        records.install()
    hook = StepHook(teacher=teacher, execute=args.execute, episode_label=label,
                    task_suite=args.task_suite, steps=JsonlWriter(out_dir / "steps.jsonl"),
                    teacher_log=JsonlWriter(out_dir / "teacher.jsonl"), records=records,
                    sampler_advance=args.sampler_advance, stash=stash, out_dir=out_dir)
    evaluator.runtime_env_step = hook

    try:
        run_cell(args, evaluator, hook, report)
    except Exception as exc:
        fail_cell_atomically(report, exc)
        traceback.print_exc()
    finally:
        env = report.pop("_env", None)
        process = report.pop("_managed_process", None)
        if env is not None:
            env.close()
        evaluator._stop_managed_policy_server(process, timeout_sec=10.0)
        if records is not None:
            report["records"] = records.status()
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report.get(k) for k in ("status", "successes", "metric_decomposition")},
                     sort_keys=True))
    if records is not None:
        # result.json is final at this point; the committed episode stores its checksum
        commit = records.finalize(args.output, report["status"])
        (out_dir / "records_commit.json").write_text(json.dumps(commit, indent=2, sort_keys=True) + "\n")
        print("RECORDS " + json.dumps(commit, sort_keys=True))
        if report["status"] == "pass" and not commit["committed"]:
            sys.exit("RECORDS_NOT_COMMITTED: the rollout passed but its learning records did not")
    if report["status"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
