"""Fail-closed identity and architecture gates for every rollout cell."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def assert_task_identity(cfg: Any, task: Any, task_id: int) -> dict[str, Any]:
    """Prove configured and loaded suite/level/task identity agree."""
    configured_suite = str(cfg.task_suite_name)
    configured_level = int(cfg.task_level)
    configured_task_id = int(task_id)
    actual_suite = str(getattr(task, "problem_folder", ""))
    actual_level = int(getattr(task, "level", -1))
    actual_task_id = int(getattr(task, "level_id", -1))

    assert actual_suite == configured_suite, (
        f"task suite mismatch: configured={configured_suite!r}, task={actual_suite!r}"
    )
    assert actual_level == configured_level, (
        f"task level mismatch: configured={configured_level}, task={actual_level}"
    )
    assert actual_task_id == configured_task_id, (
        f"task id mismatch: configured={configured_task_id}, task={actual_task_id}"
    )
    return {
        "configured_suite": configured_suite,
        "configured_level": configured_level,
        "configured_task_id": configured_task_id,
        "task_problem_folder": actual_suite,
        "task_level": actual_level,
        "task_level_id": actual_task_id,
        "task_name": str(getattr(task, "name", "")),
        "task_bddl_file": str(getattr(task, "bddl_file", "")),
        "identity_gate": "pass",
    }


def load_task_and_states(suite: Any, cfg: Any, task_id: int):
    """Load both task and states through the configured level, then gate identity."""
    level = int(cfg.task_level)
    task = suite.get_task_by_level_id(level, int(task_id))
    initial_states = suite.get_task_init_states(level, int(task_id))
    identity = assert_task_identity(cfg, task, task_id)
    return task, initial_states, identity


def assert_report_identity(report: dict[str, Any], cfg: Any, task_id: int) -> None:
    """Reject any drift between the executed identity and serialized result."""
    identity = report.get("task_identity") or {}
    expected = (
        str(cfg.task_suite_name),
        int(cfg.task_level),
        int(task_id),
        "pass",
    )
    observed = (
        identity.get("configured_suite"),
        identity.get("configured_level"),
        identity.get("configured_task_id"),
        identity.get("identity_gate"),
    )
    assert observed == expected, (
        f"result identity mismatch: expected={expected!r}, observed={observed!r}"
    )
    assert identity.get("task_problem_folder") == expected[0], "result identity suite drift"
    assert identity.get("task_level") == expected[1], "result identity level drift"
    assert identity.get("task_level_id") == expected[2], "result identity task id drift"


# _METADATA keys are full tuple reprs, e.g. "('params', 'time_mlp_in', 'kernel', 'value')".
# Signatures are matched as substrings against those reprs joined together, so each entry
# carries its surrounding quotes: bare "time_mlp_in" is a substring of "action_time_mlp_in"
# and would make every Pi0 checkpoint look like it carried Pi0.5's layers.
#
# Real diff of the two Arena checkpoints' params/_METADATA: 62 keys in Pi0.5, 61 in Pi0,
# 52 shared, 10 unique to Pi0.5, 9 unique to Pi0. Pi0.5 modulates its norms through a
# Dense_0 (adaRMS) where Pi0 uses a plain RMSNorm scale; Pi0.5 names its time MLP
# time_mlp_*, Pi0 names it action_time_mlp_* and adds a state_proj that Pi0.5 lacks.
PI05_SIGNATURES = (
    "'final_norm_1', 'Dense_0'",
    "'pre_attention_norm_1', 'Dense_0'",
    "'pre_ffw_norm_1', 'Dense_0'",
    "'time_mlp_in'",
    "'time_mlp_out'",
)
PI0_SIGNATURES = (
    "'final_norm_1', 'scale'",
    "'pre_attention_norm_1', 'scale'",
    "'pre_ffw_norm_1', 'scale'",
    "'action_time_mlp_in'",
    "'action_time_mlp_out'",
    "'state_proj'",
)
ARCHITECTURE_SIGNATURES = {
    "pi05": PI05_SIGNATURES,
    "pi0": PI0_SIGNATURES,
}


def assert_architecture(train_config, metadata_path: str | Path) -> dict:
    """Reject a config/checkpoint mismatch before policy startup, for Pi0.5 or Pi0.

    Dispatches on the config's own pi05 flag rather than a model_type string, then requires
    BOTH that the family's signature keys are present AND that the other family's are absent.
    The mutual-exclusion half is what makes this a real check: the two checkpoints share 41 of
    ~50 keys, so presence alone would pass a Pi0 checkpoint under several Pi0.5 signatures.
    """
    model = train_config.model
    is_pi05 = bool(getattr(model, "pi05", False))
    model_type = model.model_type.value
    assert model_type in ("pi0", "pi05"), f"unsupported model type {model_type!r} (pi0.5 and pi0 only)"
    family = "pi05" if is_pi05 else "pi0"
    assert (model_type == "pi05") == is_pi05, (
        f"config {train_config.name} is internally inconsistent: "
        f"model_type={model_type} but pi05={is_pi05}"
    )

    tree = json.loads(Path(metadata_path).read_text())["tree_metadata"]
    joined = "\n".join(tree)
    expected = ARCHITECTURE_SIGNATURES[family]
    other = ARCHITECTURE_SIGNATURES["pi0" if family == "pi05" else "pi05"]

    missing = [signature for signature in expected if signature not in joined]
    assert not missing, f"{family} checkpoint signature keys missing: {missing}"
    intruding = [signature for signature in other if signature in joined]
    assert not intruding, (
        f"config {train_config.name} declares {family} but the checkpoint at {metadata_path} "
        f"carries foreign signature keys: {intruding}"
    )

    return {
        "config_name": train_config.name,
        "architecture_family": family,
        "model_type": model_type,
        "pi05": is_pi05,
        "action_horizon": int(model.action_horizon),
        "signature_keys_present": True,
        "signature_keys": list(expected),
        "foreign_keys_absent": list(other),
        "loader_validation": "BaseModelConfig.load full pytree keys+shapes (check_shapes=True)",
    }
