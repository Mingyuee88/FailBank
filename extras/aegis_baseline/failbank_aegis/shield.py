"""TeacherShield wrapper around the published AEGIS shield (see vlsa_port.py).

AEGIS perceives once per episode (GLM-4.5V names the most likely obstruction,
GroundingDINO + RGB-D build its point cloud, an ellipsoid is fitted) and solves its CBF-QP
every step. Run it with ``--execute teacher``: it is the shield-in-the-loop baseline, and its
actions are what the environment executes.
"""
from __future__ import annotations

from typing import Sequence

from failbank.runtime.arena_hooks import unwrap_arena_env
from failbank.teacher.base import TeacherContext, TeacherProposal


class VlsaAegisTeacher:
    name = "vlsa_aegis"
    # the patched evaluator adds the backview camera and depth planes for this teacher
    needs_aegis_cameras = True

    def __init__(self) -> None:
        self._dino = None
        self._shield = None

    def reset(self, ctx: TeacherContext) -> None:
        from failbank_aegis import vlsa_port

        if self._dino is None:
            self._dino = vlsa_port.load_dino()
        self._shield = vlsa_port.VlsaAegisShield(
            env=unwrap_arena_env(ctx.env), task_description=ctx.task_description,
            suite_name=ctx.task_suite, out_dir=ctx.out_dir, dino_model=self._dino,
            episode_label=ctx.episode_label)
        self._shield.perceive(ctx.obs)

    def propose(self, ctx: TeacherContext, nominal: Sequence[float]) -> TeacherProposal:
        out = [float(x) for x in self._shield.safe_action(ctx.obs, list(nominal))]
        last = self._shield.last or {}
        triggered = bool(last.get("triggered"))
        return TeacherProposal(
            action=out, triggered=triggered, norm=float(last.get("norm") or 0.0),
            projection={"translation": out[:3], "min_barrier": last.get("h"),
                        "active_constraints": int(last.get("active_constraints") or 0),
                        "status": "projected" if triggered else "nominal_safe"},
            metadata={"dof": self._shield.dof})

    def audit(self) -> dict:
        """Evidence that perception and the QP actually ran (logged at the end of the cell)."""
        return dict(self._shield.audit) if self._shield is not None else {}
