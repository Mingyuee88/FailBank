"""Goal-region extraction and signed progress capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


class GoalExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class GoalRegion:
    object_name: str
    center: np.ndarray
    extent: np.ndarray

    def as_json(self) -> dict[str, Any]:
        return {
            "object": self.object_name,
            "region_center": self.center.tolist(),
            "region_extent": self.extent.tolist(),
        }


class ProgressCapture:
    def __init__(
        self,
        *,
        extract_goal: Callable[[Any], Mapping[str, Any]],
        signed_distance_to_region: Callable[..., float],
    ) -> None:
        self._extract_goal = extract_goal
        self._signed_distance_to_region = signed_distance_to_region
        self._goal: GoalRegion | None = None

    def reset(self, env: Any) -> GoalRegion:
        try:
            raw = self._extract_goal(env)
            object_name = str(raw["object"])
            center = np.asarray(raw["region_center"], dtype=np.float64).reshape(-1)
            extent = np.asarray(raw["region_extent"], dtype=np.float64).reshape(-1)
        except Exception as exc:
            self._goal = None
            raise GoalExtractionError("extract_goal returned an invalid goal") from exc

        if not object_name or center.shape != (2,) or extent.shape != (2,):
            self._goal = None
            raise GoalExtractionError(
                f"Invalid goal object/region: object={object_name!r}, "
                f"center={center.shape}, extent={extent.shape}"
            )
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(extent)):
            self._goal = None
            raise GoalExtractionError("Goal region contains non-finite values")
        if np.any(extent <= 0):
            self._goal = None
            raise GoalExtractionError("Goal region extent must be positive")

        self._goal = GoalRegion(object_name, center, extent)
        return self._goal

    @property
    def goal(self) -> GoalRegion:
        if self._goal is None:
            raise GoalExtractionError("Goal was not initialized")
        return self._goal

    def read(self, observation: Mapping[str, Any]) -> float:
        goal = self.goal
        key = f"{goal.object_name}_pos"
        try:
            object_position = np.asarray(
                observation[key],
                dtype=np.float64,
            ).reshape(-1)
        except Exception as exc:
            raise GoalExtractionError(f"Missing/invalid observation key: {key}") from exc

        if object_position.size < 2 or not np.all(np.isfinite(object_position)):
            raise GoalExtractionError(f"Invalid object position at {key}")

        # signed_distance_to_region signature is (pos_xy, center, extent) -> planar.
        value = self._signed_distance_to_region(
            object_position[:2],
            goal.center,
            goal.extent,
        )

        value = float(value)
        if not np.isfinite(value):
            raise GoalExtractionError("Progress value is non-finite")
        return value
