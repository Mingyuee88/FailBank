"""Canonical schema and hashing for Phase-1 learning records."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

SCHEMA_VERSION = "se_hwm_learning_record/v1"


def _normalize_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite float is not serializable: {value!r}")
    return 0.0 if value == 0.0 else value


def normalize_json(value: Any) -> Any:
    """Convert supported values to deterministic, JSON-compatible objects."""
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)

    if isinstance(value, np.ndarray):
        return normalize_json(value.tolist())
    if isinstance(value, np.generic):
        return normalize_json(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): normalize_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_json(item) for item in value]
    if isinstance(value, float):
        return _normalize_float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        normalize_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


@dataclass(frozen=True)
class ProjectionCandidateRaw:
    name: str
    translation: list[float]
    full_action: list[float]
    action_delta_norm: float
    min_barrier: Optional[float]
    active_constraints: Optional[int]
    status: str
    qp_executed: bool
    reused_formal_result: bool
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawStep:
    schema_version: str
    episode_id: str
    step_index: int
    infer_call_index: Optional[int]
    task_description: str

    raw_observation_refs: dict[str, dict[str, Any]]
    model_input_refs: dict[str, dict[str, Any]]
    nominal_action_chunk_ref: Optional[dict[str, Any]]

    nominal_action: list[float]
    executed_action: list[float]
    formal_projection: dict[str, Any]
    projection_candidates: list[ProjectionCandidateRaw]

    progress_pre: float
    progress_post: float
    progress_delta: float
    goal: dict[str, Any]
    object_positions: dict[str, list[float]]

    reward: float
    done: bool
    runtime_info: dict[str, Any]
    recorder_diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeRaw:
    schema_version: str
    episode_id: str
    task_description: str
    seed: Optional[int]
    runtime_method: str
    config: dict[str, Any]
    goal: dict[str, Any]
    started_at_utc: str
    steps_file: str = "raw_steps.jsonl"
    step_count: int = 0
    result_ref: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DerivedRecord:
    schema_version: str
    record_id: str
    episode_id: str
    step_index: int
    raw_step_sha256: str
    teacher_candidate: Optional[str]
    teacher_action: Optional[list[float]]
    teacher_provenance: dict[str, Any]
    outcome_window: dict[str, Any]
    record_weight: Optional[float]
    split: Optional[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def to_json_dict(value: Any) -> dict[str, Any]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise TypeError("Top-level schema object must serialize as a JSON object")
    return normalized


def to_json_line(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def validate_raw_step(step: RawStep) -> None:
    if step.schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unexpected schema: {step.schema_version}")
    if step.step_index < 0:
        raise ValueError("step_index must be non-negative")
    if len(step.nominal_action) != len(step.executed_action):
        raise ValueError("nominal_action and executed_action lengths differ")

    # The builder needs the identity candidate first and the teacher's own proposal
    # second (looked up by name). The paper's teacher adds three shadow QPs after those
    # ("late_min", "late_soft", "eef_wide"); other teachers may add their own or none.
    names = [candidate.name for candidate in step.projection_candidates]
    if names[:2] != ["nominal", "late_margin"] or len(set(names)) != len(names):
        raise ValueError(f"Candidate order mismatch: {names!r}")
