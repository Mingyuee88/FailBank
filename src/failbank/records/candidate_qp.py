"""Phase-1 projection-candidate registry and shadow-QP evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .schema_v1 import ProjectionCandidateRaw

ProjectTranslation = Callable[..., Any]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    alpha: float
    max_translation: float
    eef_radius: float
    oracle_radius: float
    obstacle_margin: float
    run_qp: bool = True


CANDIDATE_REGISTRY: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        name="nominal",
        alpha=0.0,
        max_translation=0.0,
        eef_radius=0.0,
        oracle_radius=0.04,
        obstacle_margin=0.0,
        run_qp=False,
    ),
    CandidateSpec(
        name="late_margin",
        alpha=3.0,
        max_translation=1.0,
        eef_radius=0.03,
        oracle_radius=0.04,
        obstacle_margin=0.02,
    ),
    CandidateSpec(
        name="late_min",
        alpha=3.0,
        max_translation=1.0,
        eef_radius=0.03,
        oracle_radius=0.04,
        obstacle_margin=0.0,
    ),
    CandidateSpec(
        name="late_soft",
        alpha=1.0,
        max_translation=1.0,
        eef_radius=0.03,
        oracle_radius=0.04,
        obstacle_margin=0.0,
    ),
    CandidateSpec(
        name="eef_wide",
        alpha=3.0,
        max_translation=1.0,
        eef_radius=0.06,
        oracle_radius=0.04,
        obstacle_margin=0.0,
    ),
)


def _result_to_candidate(
    spec: CandidateSpec,
    nominal_action: np.ndarray,
    result: Any,
    *,
    qp_executed: bool,
    reused_formal_result: bool,
) -> ProjectionCandidateRaw:
    full_action = nominal_action.copy()
    translation = np.asarray(result.translation, dtype=np.float64).reshape(3)
    full_action[:3] = translation
    return ProjectionCandidateRaw(
        name=spec.name,
        translation=translation.tolist(),
        full_action=full_action.tolist(),
        action_delta_norm=float(np.linalg.norm(full_action - nominal_action)),
        min_barrier=None if result.min_barrier is None else float(result.min_barrier),
        active_constraints=int(result.active_constraints),
        status=str(result.status),
        qp_executed=qp_executed,
        reused_formal_result=reused_formal_result,
        parameters={
            "alpha": spec.alpha,
            "max_translation": spec.max_translation,
            "eef_radius": spec.eef_radius,
            "oracle_radius": spec.oracle_radius,
            "obstacle_margin": spec.obstacle_margin,
        },
    )


def evaluate_projection_candidates(
    *,
    nominal_action: Sequence[float],
    eef_position: Sequence[float],
    obstacles: Sequence[Any],
    formal_result: Any,
    project_translation: ProjectTranslation,
    formal_candidate_name: str = "late_margin",
) -> list[ProjectionCandidateRaw]:
    """
    Return candidates in registry order.

    `formal_result` is reused exactly once. The other three non-nominal
    candidates execute shadow QPs. Nominal is identity/no-QP.
    """
    nominal = np.asarray(nominal_action, dtype=np.float64).reshape(-1)
    eef = np.asarray(eef_position, dtype=np.float64).reshape(-1)
    if nominal.size < 3 or eef.size < 3:
        raise ValueError("Candidate evaluation requires 3D action and EEF position")

    results: list[ProjectionCandidateRaw] = []
    for spec in CANDIDATE_REGISTRY:
        if not spec.run_qp:
            results.append(
                ProjectionCandidateRaw(
                    name=spec.name,
                    translation=nominal[:3].tolist(),
                    full_action=nominal.tolist(),
                    action_delta_norm=0.0,
                    min_barrier=None,
                    active_constraints=None,
                    status="identity/no_qp",
                    qp_executed=False,
                    reused_formal_result=False,
                    parameters={"oracle_radius": spec.oracle_radius},
                )
            )
            continue

        if spec.name == formal_candidate_name:
            candidate_result = formal_result
            reused = True
            qp_executed = False
        else:
            candidate_result = project_translation(
                nominal[:3],
                eef[:3],
                obstacles,
                alpha=spec.alpha,
                max_translation=spec.max_translation,
                eef_radius=spec.eef_radius,
                obstacle_margin=spec.obstacle_margin,
            )
            reused = False
            qp_executed = True

        results.append(
            _result_to_candidate(
                spec,
                nominal,
                candidate_result,
                qp_executed=qp_executed,
                reused_formal_result=reused,
            )
        )

    return results
