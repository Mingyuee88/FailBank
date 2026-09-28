"""The teacher interface: an observe-only shield that labels, never acts.

A teacher sees the state and the policy's nominal action a_t and returns a proposal
~a_t together with a trigger bit z_t. During FailBank collection the environment always
executes a_t; ~a_t is only written into the learning record. That is what lets the
collection reach the hazard states the policy actually visits (a shield in the loop keeps
the trajectory away from them, so no failure is ever recorded).

Anything that can produce (~a_t, z_t) from the current state can be plugged in:

    class MyShield:
        name = "my_shield"
        def reset(self, ctx: TeacherContext) -> None: ...
        def propose(self, ctx: TeacherContext, nominal: Sequence[float]) -> TeacherProposal: ...

and selected with ``failbank-rollout --teacher my_package.my_module:MyShield``.

Replacing the teacher changes the labels, so it will not reproduce the paper's numbers;
that is expected. The paper's teacher is ``failbank.teacher.oracle_geometry``.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

# Name under which the teacher's own proposal is stored in `projection_candidates`.
# The record builder looks the formal candidate up by this name (`teacher_quality`
# reads its `action_delta_norm`), so it is part of the record schema. The string is
# historical: it names the CBF parameter set the paper's teacher uses.
FORMAL_CANDIDATE = "late_margin"


@dataclass(frozen=True)
class TeacherContext:
    """Everything a teacher may read at one control step."""

    env: Any                      # the Arena environment (simulator access is allowed)
    obs: Mapping[str, Any]        # the observation the policy acted on
    features: Mapping[str, Any]   # cost-relation features (see runtime.arena_hooks)
    step: int                     # control-step index within the episode
    task_description: str = ""
    task_suite: str = ""
    out_dir: str = "."            # the cell's output directory (for teacher artefacts)
    episode_label: str = ""


@dataclass(frozen=True)
class TeacherProposal:
    """What a teacher says about one nominal action.

    action      full 7-D proposal ~a_t (the learning target when triggered)
    triggered   z_t: the teacher would have intervened on this step
    projection  written verbatim into the record's ``formal_projection``; must carry
                ``translation``, ``min_barrier``, ``active_constraints`` and ``status``
    norm        ||~a_t[:3] - a_t[:3]||, logged to telemetry
    """

    action: list[float]
    triggered: bool
    projection: dict[str, Any]
    norm: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TeacherShield(Protocol):
    name: str

    def reset(self, ctx: TeacherContext) -> None:
        """Called once at the first step of every episode."""

    def propose(self, ctx: TeacherContext, nominal: Sequence[float]) -> TeacherProposal:
        """Label one nominal action. Must not modify the environment."""


def default_candidates(nominal: Sequence[float], proposal: TeacherProposal) -> list[Any]:
    """Candidate list for teachers that do not provide their own.

    Two entries: the identity (the policy's own action) and the teacher's proposal under
    FORMAL_CANDIDATE. This is the minimum the record builder needs.
    """
    import numpy as np

    from failbank.records.schema_v1 import ProjectionCandidateRaw

    nom = np.asarray(nominal, dtype=np.float64).reshape(-1)
    act = np.asarray(proposal.action, dtype=np.float64).reshape(-1)
    p = proposal.projection
    return [
        ProjectionCandidateRaw(
            name="nominal", translation=nom[:3].tolist(), full_action=nom.tolist(),
            action_delta_norm=0.0, min_barrier=None, active_constraints=None,
            status="identity/no_qp", qp_executed=False, reused_formal_result=False),
        ProjectionCandidateRaw(
            name=FORMAL_CANDIDATE, translation=act[:3].tolist(), full_action=act.tolist(),
            action_delta_norm=float(np.linalg.norm(act - nom)),
            min_barrier=p.get("min_barrier"), active_constraints=p.get("active_constraints"),
            status=str(p.get("status", "")), qp_executed=False, reused_formal_result=True),
    ]


def load_teacher(spec: str, **oracle_config: Any) -> TeacherShield | None:
    """Resolve ``--teacher``: ``none``, ``oracle_geometry``, or ``package.module:Class``.

    ``oracle_config`` (the CBF constants) only applies to the built-in teacher; a custom
    class is constructed without arguments and configures itself.
    """
    if spec in ("", "none"):
        return None
    if spec == "oracle_geometry":
        from failbank.teacher.oracle_geometry import OracleGeometryTeacher
        return OracleGeometryTeacher(**oracle_config)
    if ":" not in spec:
        raise ValueError(f"--teacher must be none, oracle_geometry or module:Class, got {spec!r}")
    module_name, _, attr = spec.partition(":")
    cls = getattr(importlib.import_module(module_name), attr)
    teacher = cls()
    if not isinstance(teacher, TeacherShield):
        raise TypeError(f"{spec} does not implement TeacherShield (needs name/reset/propose)")
    return teacher
