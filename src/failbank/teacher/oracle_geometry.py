# Projection adapted from vlsa-aegis (MIT, Copyright (c) 2023 Lifelong Robot Learning),
# https://github.com/THU-RCSCT/vlsa-aegis
"""The paper's teacher: a CBF projection onto simulator-privileged obstacle geometry.

Sequential affine CBF projection migrated from vlsa_aegis.cbf_qp, applied to spheres
placed on the BDDL cost-state objects (cost-state objects minus the task's objects of
interest) read directly from the simulator. No perception, no VLM: this teacher only
works in simulation.

Behaviour is kept bit-identical to the implementation that produced the paper's records.
Per step:

    obstacles   = spheres(radius=oracle_radius) at every cost-state hazard body
    translation = project(clip(a_t[:3], +-max_translation), eef, obstacles)
    ~a_t        = a_t with [:3] replaced by translation
    z_t         = status == "projected" and ||~a_t[:3] - a_t[:3]|| > override_eps

Note that ~a_t[:3] is the *clipped* nominal when no constraint is active, so a nominal
translation outside +-max_translation gives ~a_t != a_t with z_t = False.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from failbank.teacher.base import TeacherContext, TeacherProposal


# --------------------------------------------------------------------------- geometry

@dataclass(frozen=True)
class Ellipsoid:
    center: np.ndarray
    radii: np.ndarray
    rotation: np.ndarray
    label: str = ""

    @classmethod
    def sphere(cls, center: Sequence[float], radius: float, label: str = ""):
        return cls(np.asarray(center, dtype=float), np.full(3, float(radius)), np.eye(3), label)

    def support_radius(self, direction: np.ndarray) -> float:
        direction = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            return float(np.max(self.radii))
        return float(np.linalg.norm(self.radii * (self.rotation.T @ (direction / norm))))


def _flatten_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _flatten_strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _flatten_strings(item)


def _find_base_env(env: Any) -> Any | None:
    current = env
    for _ in range(8):
        if current is None:
            return None
        if getattr(current, "sim", None) is not None and isinstance(getattr(current, "obj_body_id", None), dict):
            return current
        current = getattr(current, "env", None)
    return None


def extract_oracle_ellipsoids(env: Any, *, default_radius: float,
                              selected_labels: Sequence[str] | None = None) -> list[Ellipsoid]:
    """Spheres on the BDDL cost-state objects, excluding the task's objects of interest."""
    base = _find_base_env(env)
    if base is None:
        return []
    parsed = getattr(base, "parsed_problem", {}) or {}
    cost = set(_flatten_strings(parsed.get("cost_state", [])))
    task = set(parsed.get("obj_of_interest", []))
    names = set(base.obj_body_id)
    candidates = sorted((cost & names) - task)
    selectors = [str(x).lower().replace(" ", "_") for x in selected_labels or []]
    if selectors:
        candidates = [x for x in candidates if any(s in x.lower() or x.lower() in s for s in selectors)]
    result = []
    for name in candidates:
        body_id = base.obj_body_id[name]
        try:
            center = np.asarray(base.sim.data.body_xpos[body_id], dtype=float).copy()
        except Exception:
            continue
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            continue
        result.append(Ellipsoid.sphere(center, default_radius, name))
    return result


# --------------------------------------------------------------------------- projection

@dataclass(frozen=True)
class ProjectionResult:
    translation: np.ndarray
    status: str
    min_barrier: float | None
    active_constraints: int


def project_translation(nominal: Sequence[float], eef_position: Sequence[float],
                        obstacles: Sequence[Ellipsoid], *, alpha: float, max_translation: float,
                        eef_radius: float, obstacle_margin: float) -> ProjectionResult:
    """One sequential pass of affine CBF half-space projections (vlsa_aegis.cbf_qp)."""
    velocity = np.clip(np.asarray(nominal, dtype=float), -max_translation, max_translation)
    eef = np.asarray(eef_position, dtype=float)
    min_barrier = None
    active = 0
    for obstacle in obstacles:
        away = eef - obstacle.center
        distance = float(np.linalg.norm(away))
        normal = np.array([1., 0., 0.]) if distance <= 1e-12 else away / distance
        clearance = eef_radius + obstacle.support_radius(normal) + obstacle_margin
        barrier = distance - clearance
        min_barrier = barrier if min_barrier is None else min(min_barrier, barrier)
        lower = -alpha * barrier
        current = float(normal @ velocity)
        if current < lower:
            velocity = np.clip(velocity + (lower - current) * normal, -max_translation, max_translation)
            active += 1
    return ProjectionResult(velocity, "projected" if active else "nominal_safe", min_barrier, active)


# --------------------------------------------------------------------------- teacher

@dataclass(frozen=True)
class OracleGeometryConfig:
    # Defaults are the values every paper run used (not the historical code defaults).
    alpha: float = 3.0
    max_translation: float = 1.0
    eef_radius: float = 0.03
    oracle_radius: float = 0.04
    obstacle_margin: float = 0.02
    override_eps: float = 1e-6

    def validate(self) -> None:
        if self.alpha <= 0 or self.max_translation <= 0 or self.oracle_radius <= 0:
            raise ValueError("positive CBF scales required")
        if self.eef_radius < 0 or self.obstacle_margin < 0 or self.override_eps < 0:
            raise ValueError("invalid CBF radius/margin/epsilon")


class OracleGeometryTeacher:
    """CBF teacher on privileged simulator geometry (the paper's default)."""

    name = "oracle_geometry"

    def __init__(self, **config: float) -> None:
        self.cfg = OracleGeometryConfig(**config)
        self.cfg.validate()
        # kept for candidate evaluation of the step just proposed
        self._last: dict[str, Any] = {}

    def reset(self, ctx: TeacherContext) -> None:
        self._last = {}

    def propose(self, ctx: TeacherContext, nominal: Sequence[float]) -> TeacherProposal:
        cfg = self.cfg
        obstacles = extract_oracle_ellipsoids(ctx.env, default_radius=cfg.oracle_radius)
        eef = list(ctx.features.get("eef_position") or [])
        nominal = [float(x) for x in nominal]
        executed = list(nominal)
        result = None
        norm = 0.0
        if len(eef) >= 3:
            result = project_translation(nominal[:3], eef[:3], obstacles, alpha=cfg.alpha,
                                         max_translation=cfg.max_translation,
                                         eef_radius=cfg.eef_radius,
                                         obstacle_margin=cfg.obstacle_margin)
            executed[:3] = [float(x) for x in result.translation]
            norm = sum((executed[i] - nominal[i]) ** 2 for i in range(3)) ** .5
        triggered = bool(result is not None and result.status == "projected"
                         and norm > cfg.override_eps)
        self._last = {"eef": eef, "obstacles": obstacles, "result": result}
        if result is None:
            projection: dict[str, Any] = {}
        else:
            projection = {
                "translation": [float(x) for x in result.translation],
                "min_barrier": None if result.min_barrier is None else float(result.min_barrier),
                "active_constraints": int(result.active_constraints),
                "status": str(result.status),
            }
        return TeacherProposal(action=executed, triggered=triggered, projection=projection,
                               norm=float(norm),
                               metadata={"obstacle_count": len(obstacles)})

    def candidates(self, nominal: Sequence[float], proposal: TeacherProposal):
        """The five-candidate registry the paper's records carry (see records.candidate_qp)."""
        from failbank.records.candidate_qp import evaluate_projection_candidates

        last = self._last
        if last.get("result") is None:
            raise RuntimeError("formal projection unavailable; fail-closed")
        return evaluate_projection_candidates(
            nominal_action=nominal, eef_position=last["eef"], obstacles=last["obstacles"],
            formal_result=last["result"], project_translation=project_translation,
            formal_candidate_name="late_margin")
