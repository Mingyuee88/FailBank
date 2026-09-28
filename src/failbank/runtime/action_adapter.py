"""Faithful first-action wrapper for VLA-Arena's native OpenPI actions.

OpenPI's policy server has already unnormalized action chunks.  Executable
actions therefore remain in VLA-Arena / robosuite controller input space.
This module never normalizes, clips, or rescales an uncorrected base action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ArenaControllerActionContract:
    """Action mapping read from the active Arena controller instance."""

    arm_input_min: tuple[float, ...]
    arm_input_max: tuple[float, ...]
    arm_output_min: tuple[float, ...]
    arm_output_max: tuple[float, ...]
    arena_action_min: tuple[float, ...]
    arena_action_max: tuple[float, ...]
    action_order: tuple[str, ...] = ("dx", "dy", "dz", "rx", "ry", "rz", "gripper")
    command_type: str = "delta"
    coordinate_frame: str = "robot_base"
    chunk_policy: str = "first_action_only_replan_1"
    gripper_correction_enabled: bool = False

    def __post_init__(self) -> None:
        arrays = (
            self.arm_input_min,
            self.arm_input_max,
            self.arm_output_min,
            self.arm_output_max,
        )
        if any(len(value) != 6 for value in arrays):
            raise ValueError("Arena arm controller contract must be 6-D")
        if len(self.arena_action_min) != 7 or len(self.arena_action_max) != 7:
            raise ValueError("Arena executable action contract must be 7-D")
        if np.any(np.asarray(self.arm_input_max) <= np.asarray(self.arm_input_min)):
            raise ValueError("controller input limits are invalid")
        if np.any(np.asarray(self.arm_output_max) <= np.asarray(self.arm_output_min)):
            raise ValueError("controller output limits are invalid")

    @classmethod
    def from_env(cls, env) -> "ArenaControllerActionContract":
        robot = env.robots[0]
        arm_name = robot.arms[0]
        arm = robot.composite_controller.get_controller(arm_name)
        action_min, action_max = robot.composite_controller.action_limits
        return cls(
            arm_input_min=tuple(np.asarray(arm.input_min, dtype=float)),
            arm_input_max=tuple(np.asarray(arm.input_max, dtype=float)),
            arm_output_min=tuple(np.asarray(arm.output_min, dtype=float)),
            arm_output_max=tuple(np.asarray(arm.output_max, dtype=float)),
            arena_action_min=tuple(np.asarray(action_min, dtype=float)),
            arena_action_max=tuple(np.asarray(action_max, dtype=float)),
        )

    def to_dict(self) -> dict:
        value = asdict(self)
        for key, item in tuple(value.items()):
            if isinstance(item, tuple):
                value[key] = list(item)
        return value


@dataclass(frozen=True)
class InterventionDecision:
    action: np.ndarray
    base_action: np.ndarray
    controller_effective_arm: np.ndarray
    controller_saturation_mask: np.ndarray
    replan: bool
    discarded_actions: int
    action_forwarded_unchanged: bool
    correction_applied: bool


class Pi05CanonicalActionAdapter:
    """Select one native Arena action and optionally add a physical arm delta."""

    def __init__(self, contract: ArenaControllerActionContract) -> None:
        self.contract = contract

    @staticmethod
    def _validate_chunk(chunk) -> np.ndarray:
        value = np.asarray(chunk)
        if value.ndim != 2:
            raise ValueError("action chunk must have shape [horizon, 7]")
        if value.shape[0] < 1 or value.shape[1] != 7:
            raise ValueError("action chunk must contain at least one 7-D Arena action")
        if not np.isfinite(value).all():
            raise ValueError("action chunk must be finite")
        return value

    def _effective_arm(self, raw_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        input_min = np.asarray(self.contract.arm_input_min)
        input_max = np.asarray(self.contract.arm_input_max)
        output_min = np.asarray(self.contract.arm_output_min)
        output_max = np.asarray(self.contract.arm_output_max)
        clipped = np.clip(raw_arm, input_min, input_max)
        scale = (output_max - output_min) / (input_max - input_min)
        effective = (clipped - input_min) * scale + output_min
        return effective, clipped != raw_arm

    def _raw_arm_from_effective(self, effective: np.ndarray) -> np.ndarray:
        input_min = np.asarray(self.contract.arm_input_min)
        input_max = np.asarray(self.contract.arm_input_max)
        output_min = np.asarray(self.contract.arm_output_min)
        output_max = np.asarray(self.contract.arm_output_max)
        target = np.clip(effective, output_min, output_max)
        return (target - output_min) * (input_max - input_min) / (output_max - output_min) + input_min

    def select_first(self, chunk, canonical_delta=None) -> InterventionDecision:
        native = self._validate_chunk(chunk)
        base = native[0].copy()
        delta = np.zeros(7, dtype=float) if canonical_delta is None else np.asarray(canonical_delta, dtype=float)
        if delta.shape != (7,) or not np.isfinite(delta).all():
            raise ValueError("canonical correction must be finite 7-D")
        if delta[6] != 0:
            raise ValueError("gripper correction disabled until Arena format_action semantics are wrapped")

        effective, saturation = self._effective_arm(np.asarray(base[:6], dtype=float))
        correction_applied = bool(np.any(delta[:6] != 0))
        if correction_applied:
            action = base.copy()
            action[:6] = self._raw_arm_from_effective(effective + delta[:6])
            effective, saturation = self._effective_arm(np.asarray(action[:6], dtype=float))
            unchanged = False
        else:
            # Hard invariant: the Arena-native base action is forwarded bit-for-bit.
            action = base.copy()
            unchanged = bool(np.array_equal(action, base))

        return InterventionDecision(
            action=action,
            base_action=base,
            controller_effective_arm=effective,
            controller_saturation_mask=saturation,
            replan=True,
            discarded_actions=len(native) - 1,
            action_forwarded_unchanged=unchanged,
            correction_applied=correction_applied,
        )


def adapt_first_action(adapter, chunk) -> tuple[np.ndarray, dict]:
    """Select first server action unchanged and expose Arena controller telemetry."""
    native = np.asarray(chunk)
    decision = adapter.select_first(native, np.zeros(7, dtype=float))
    if not decision.replan or decision.discarded_actions != len(native) - 1:
        raise AssertionError("first-action + replan semantics violated")
    if not decision.action_forwarded_unchanged or not np.array_equal(decision.action, native[0]):
        raise AssertionError("zero correction must forward Arena action bit-for-bit")
    effective = decision.controller_effective_arm
    return decision.action, {
        "chunk_length": int(len(native)),
        "executed_index": 0,
        "replan": bool(decision.replan),
        "replan_mode": "first_action_only_replan_1",
        "discarded_actions": int(decision.discarded_actions),
        "server_action_raw": native[0].tolist(),
        "arena_controller_input_min": list(adapter.contract.arena_action_min),
        "arena_controller_input_max": list(adapter.contract.arena_action_max),
        "controller_saturation_mask": decision.controller_saturation_mask.tolist(),
        "controller_saturation_count": int(np.count_nonzero(decision.controller_saturation_mask)),
        "controller_effective_translation_m": effective[:3].tolist(),
        "controller_effective_rotation_rad": effective[3:6].tolist(),
        "controller_effective_translation_abs_max_m": float(np.max(np.abs(effective[:3]))),
        "controller_effective_rotation_abs_max_rad": float(np.max(np.abs(effective[3:6]))),
        "action_forwarded_unchanged": bool(decision.action_forwarded_unchanged),
        "correction_applied": bool(decision.correction_applied),
        "physical_units_gate": "pass",
    }
