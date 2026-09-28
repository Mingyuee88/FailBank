"""Semantic goal extraction and planar progress to the goal region (used by Stage 1 records)."""

from __future__ import annotations

import math
from typing import Any


class GoalExtractionError(RuntimeError):
    """Raised when goal semantics or region geometry cannot be bound exactly."""


def _name(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    for attribute in ("name", "object_name", "site_name", "body_name"):
        candidate = getattr(value, attribute, None)
        if isinstance(candidate, str) and candidate:
            return candidate
    if isinstance(value, dict):
        for key in ("name", "object", "object_name", "region", "region_name"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
    return None


def _predicate_parts(node: Any) -> tuple[str, list[str]] | None:
    """Normalize common parsed-problem predicate representations."""
    if isinstance(node, str):
        text = node.strip().strip("()")
        tokens = text.replace(",", " ").split()
        if tokens:
            return tokens[0].lower(), tokens[1:]
        return None

    if isinstance(node, (list, tuple)) and node:
        predicate = _name(node[0]) or str(node[0])
        arguments = []
        for item in node[1:]:
            item_name = _name(item)
            arguments.append(item_name if item_name is not None else str(item))
        return predicate.lower(), arguments

    if isinstance(node, dict):
        predicate = (
            node.get("predicate")
            or node.get("name")
            or node.get("relation")
            or node.get("operator")
        )
        arguments = (
            node.get("arguments")
            or node.get("args")
            or node.get("entities")
            or node.get("objects")
        )
        if isinstance(predicate, str) and isinstance(arguments, (list, tuple)):
            normalized = []
            for item in arguments:
                item_name = _name(item)
                normalized.append(
                    item_name if item_name is not None else str(item)
                )
            return predicate.lower(), normalized

        for relation in ("in", "inside", "on", "on_top_of"):
            if relation in node:
                value = node[relation]
                if isinstance(value, (list, tuple)) and len(value) >= 2:
                    return relation, [str(value[0]), str(value[1])]

    predicate = (
        getattr(node, "predicate", None)
        or getattr(node, "name", None)
        or getattr(node, "relation", None)
    )
    arguments = (
        getattr(node, "arguments", None)
        or getattr(node, "args", None)
        or getattr(node, "entities", None)
    )
    if predicate is not None and isinstance(arguments, (list, tuple)):
        predicate_name = _name(predicate) or str(predicate)
        normalized = []
        for item in arguments:
            item_name = _name(item)
            normalized.append(item_name if item_name is not None else str(item))
        return predicate_name.lower(), normalized

    return None


def _walk_goal_state(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_goal_state(child)
    elif isinstance(value, (list, tuple, set)):
        yield value
        for child in value:
            yield from _walk_goal_state(child)
    else:
        yield value


def _parsed_problem(env: Any) -> Any:
    candidates = [
        env,
        getattr(env, "env", None),
        getattr(env, "unwrapped", None),
        getattr(env, "arena_env", None),
        getattr(env, "task", None),
    ]
    for owner in candidates:
        if owner is None:
            continue
        for attribute in (
            "parsed_problem",
            "_parsed_problem",
            "problem",
            "problem_info",
        ):
            value = getattr(owner, attribute, None)
            if value is not None:
                return value
    raise GoalExtractionError(
        "parsed_problem was not found on env, env.env, env.unwrapped, "
        "env.arena_env, or env.task"
    )


def _goal_state(parsed_problem: Any) -> Any:
    if isinstance(parsed_problem, dict):
        for key in ("goal_state", "goal", "goals"):
            if key in parsed_problem:
                return parsed_problem[key]

    for attribute in ("goal_state", "goal", "goals"):
        value = getattr(parsed_problem, attribute, None)
        if value is not None:
            return value

    raise GoalExtractionError(
        "parsed_problem exists but has no goal_state/goal/goals field"
    )


def _semantic_goal(goal_state: Any) -> tuple[str, str]:
    supported = {"in", "inside", "on", "on_top_of", "ontop"}
    matches: list[tuple[str, str, str]] = []

    for node in _walk_goal_state(goal_state):
        parsed = _predicate_parts(node)
        if parsed is None:
            continue
        predicate, arguments = parsed
        predicate = predicate.lower().replace("-", "_")
        if predicate not in supported or len(arguments) < 2:
            continue
        object_name = str(arguments[0]).strip()
        region_name = str(arguments[1]).strip()
        if object_name and region_name:
            matches.append((predicate, object_name, region_name))

    unique = []
    seen = set()
    for match in matches:
        key = (match[1], match[2])
        if key not in seen:
            seen.add(key)
            unique.append(match)

    if not unique:
        raise GoalExtractionError(
            "goal_state contains no supported (in/on <object> <region>) predicate"
        )
    if len(unique) > 1:
        descriptions = ", ".join(
            f"{predicate}({object_name}, {region_name})"
            for predicate, object_name, region_name in unique
        )
        raise GoalExtractionError(
            "goal_state maps to multiple object-region goals; explicit task "
            f"selection is required: {descriptions}"
        )

    return unique[0][1], unique[0][2]


def _xy(value: Any) -> list[float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            return None

    try:
        if hasattr(value, "tolist"):
            return _xy(value.tolist())
    except Exception:
        return None

    return None


def _extent_xy(value: Any) -> list[float] | None:
    result = _xy(value)
    if result is None:
        return None
    result = [abs(result[0]), abs(result[1])]
    if result[0] <= 0.0 or result[1] <= 0.0:
        return None
    return result


def _geometry_from_object(region: Any) -> tuple[list[float], list[float]] | None:
    if region is None:
        return None

    if isinstance(region, dict):
        center = (
            region.get("region_center")
            or region.get("center")
            or region.get("position")
            or region.get("pos")
            or region.get("site_pos")
            or region.get("body_pos")
        )
        extent = (
            region.get("region_extent")
            or region.get("extent")
            or region.get("half_extent")
            or region.get("half_extents")
            or region.get("size")
            or region.get("site_size")
        )
    else:
        center = None
        extent = None
        for attribute in (
            "region_center",
            "center",
            "position",
            "pos",
            "site_pos",
            "body_pos",
        ):
            center = getattr(region, attribute, None)
            if center is not None:
                break
        for attribute in (
            "region_extent",
            "extent",
            "half_extent",
            "half_extents",
            "size",
            "site_size",
        ):
            extent = getattr(region, attribute, None)
            if extent is not None:
                break

    center_xy = _xy(center)
    extent_xy = _extent_xy(extent)
    if center_xy is not None and extent_xy is not None:
        return center_xy, extent_xy
    return None


def _geometry_from_get_region(env: Any, region_name: str):
    for owner in (
        env,
        getattr(env, "env", None),
        getattr(env, "unwrapped", None),
        getattr(env, "arena_env", None),
        getattr(env, "task", None),
    ):
        if owner is None:
            continue
        getter = getattr(owner, "get_region", None)
        if callable(getter):
            try:
                geometry = _geometry_from_object(getter(region_name))
                if geometry is not None:
                    return geometry
            except Exception:
                continue
    return None


def _geometry_from_region_maps(env: Any, region_name: str):
    for owner in (
        env,
        getattr(env, "env", None),
        getattr(env, "unwrapped", None),
        getattr(env, "arena_env", None),
        getattr(env, "task", None),
    ):
        if owner is None:
            continue
        for attribute in (
            "regions",
            "region_dict",
            "region_map",
            "goal_regions",
            "fixtures",
        ):
            mapping = getattr(owner, attribute, None)
            if not isinstance(mapping, dict):
                continue
            for candidate_name, region in mapping.items():
                if str(candidate_name) == region_name:
                    geometry = _geometry_from_object(region)
                    if geometry is not None:
                        return geometry
    return None


def _model_and_data(env: Any) -> tuple[Any, Any]:
    for owner in (
        env,
        getattr(env, "env", None),
        getattr(env, "unwrapped", None),
        getattr(env, "arena_env", None),
        getattr(env, "sim", None),
    ):
        if owner is None:
            continue
        sim = getattr(owner, "sim", None)
        if sim is not None:
            model = getattr(sim, "model", None)
            data = getattr(sim, "data", None)
            if model is not None and data is not None:
                return model, data

        model = getattr(owner, "model", None)
        data = getattr(owner, "data", None)
        if model is not None and data is not None:
            return model, data

    raise GoalExtractionError("MuJoCo model/data pair was not found")


def _named_id(model: Any, kind: str, name: str) -> int | None:
    method = getattr(model, f"{kind}_name2id", None)
    if callable(method):
        try:
            return int(method(name))
        except Exception:
            pass

    accessor = getattr(model, kind, None)
    if callable(accessor):
        try:
            item = accessor(name)
            item_id = getattr(item, "id", None)
            if item_id is not None:
                return int(item_id)
        except Exception:
            pass

    try:
        import mujoco

        object_type = (
            mujoco.mjtObj.mjOBJ_SITE
            if kind == "site"
            else mujoco.mjtObj.mjOBJ_BODY
        )
        item_id = int(mujoco.mj_name2id(model, object_type, name))
        return item_id if item_id >= 0 else None
    except Exception:
        return None


def _geometry_from_mujoco(env: Any, region_name: str):
    try:
        model, data = _model_and_data(env)
    except GoalExtractionError:
        return None

    site_id = _named_id(model, "site", region_name)
    if site_id is not None:
        try:
            center = _xy(data.site_xpos[site_id])
            extent = _extent_xy(model.site_size[site_id])
            if center is not None and extent is not None:
                return center, extent
        except Exception:
            pass

    body_id = _named_id(model, "body", region_name)
    if body_id is None:
        for owner in (env, getattr(env, "env", None), getattr(env, "unwrapped", None), getattr(env, "arena_env", None)):
            mapping = getattr(owner, "obj_body_id", None)
            if isinstance(mapping, dict) and region_name in mapping:
                try:
                    body_id = int(mapping[region_name])
                except (TypeError, ValueError):
                    body_id = None
                break
    if body_id is not None:
        try:
            positions = getattr(data, "body_xpos", getattr(data, "xpos", None))
            center = _xy(positions[body_id])
        except Exception:
            center = None

        extent = None
        try:
            geom_bodyid = model.geom_bodyid
            geom_size = model.geom_size
            matching = [
                index
                for index in range(len(geom_bodyid))
                if int(geom_bodyid[index]) == body_id
            ]
            if matching:
                extent = [
                    max(abs(float(geom_size[index][0])) for index in matching),
                    max(abs(float(geom_size[index][1])) for index in matching),
                ]
        except Exception:
            extent = None

        if center is not None and _extent_xy(extent) is not None:
            return center, _extent_xy(extent)

    return None


def extract_goal(env: Any) -> dict[str, Any]:
    """Bind parsed goal semantics to live region geometry.

    Missing or ambiguous semantic/geometry bindings are hard errors.
    """
    parsed_problem = _parsed_problem(env)
    goal_state = _goal_state(parsed_problem)
    object_name, region_name = _semantic_goal(goal_state)

    attempts = (
        _geometry_from_get_region,
        _geometry_from_region_maps,
        _geometry_from_mujoco,
    )
    geometry = None
    for attempt in attempts:
        geometry = attempt(env, region_name)
        if geometry is not None:
            break

    if geometry is None:
        raise GoalExtractionError(
            f"goal region {region_name!r} was identified semantically, but its "
            "center/extent could not be queried via get_region, region maps, "
            "MuJoCo site geometry, or MuJoCo body geometry"
        )

    center, extent = geometry
    return {
        "object": object_name,
        "region_center": [float(center[0]), float(center[1])],
        "region_extent": [float(extent[0]), float(extent[1])],
    }


def signed_distance_to_region(
    pos_xy: list[float] | tuple[float, float],
    center: list[float] | tuple[float, float],
    extent: list[float] | tuple[float, float],
) -> float:
    """Signed Euclidean distance to an axis-aligned rectangle boundary.

    The result is positive outside, zero on the boundary, and negative inside.
    ``extent`` is interpreted as the rectangle's positive half-extent.
    """
    if len(pos_xy) < 2 or len(center) < 2 or len(extent) < 2:
        raise ValueError("pos_xy, center, and extent must each have two values")

    ex = float(extent[0])
    ey = float(extent[1])
    if ex <= 0.0 or ey <= 0.0:
        raise ValueError("region extent must be strictly positive")

    dx = abs(float(pos_xy[0]) - float(center[0])) - ex
    dy = abs(float(pos_xy[1]) - float(center[1])) - ey

    outside = math.hypot(max(dx, 0.0), max(dy, 0.0))
    inside = min(max(dx, dy), 0.0)
    return outside + inside
