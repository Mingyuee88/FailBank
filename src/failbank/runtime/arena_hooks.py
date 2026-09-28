"""Read-only simulator features used by the teacher, the records and the cost metrics.

``cost_relation_features(env, obs)`` parses the task's BDDL cost predicates and returns,
among others:

    cost_predicate_values   truth value of every cost predicate at this step
    cost_pair_min_distance  min centre distance over the (object, hazard) pairs of the
                            ``incontact`` / ``checkdistance`` predicates
    cost_pair_contacts      number of those pairs currently in contact
    eef_position            end-effector position (from ``robot0_eef_pos``)

Everything here only reads simulator state; nothing steps or modifies the environment.
"""
from __future__ import annotations

import math
from typing import Any


def to_list(action: Any) -> list[float]:
    if hasattr(action, "tolist"):
        action = action.tolist()
    return [float(value) for value in action]


def cost_relation_features(env: Any, obs: Any = None) -> dict[str, Any]:
    env = unwrap_arena_env(env)
    parsed = getattr(env, "parsed_problem", {}) or {}
    cost_state = parsed.get("cost_state", []) or []
    pairs: list[tuple[str, str]] = []
    fall_objects: list[str] = []
    predicate_values: list[dict[str, Any]] = []
    preserve_pairs: list[tuple[str, str]] = []
    distance_thresholds: list[float] = []
    predicate_kinds: list[str] = []
    for state in cost_state:
        if not isinstance(state, (list, tuple)) or not state:
            continue
        predicate_values.append(_cost_predicate_value(env, state))
        predicate_kinds.append(str(state[0]))
        if state[0] == "incontact" and len(state) >= 3:
            pairs.append((str(state[1]), str(state[2])))
        elif state[0] == "checkdistance" and len(state) >= 3:
            if len(state) >= 4:
                try:
                    distance_thresholds.append(float(state[3]))
                except (TypeError, ValueError):
                    pass
            # distance-form cost predicate: (checkdistance <object> <hazard> <threshold>).
            # Same pair semantics as incontact, different suite. Note the sibling
            # `checkgripperdistance <hazard> <threshold>` names ONE object -- it is the
            # gripper-side conjunct and must not become a pair.
            pairs.append((str(state[1]), str(state[2])))
        elif state[0] == "noton" and len(state) >= 3:
            # (noton <contents> <container>): the cost fires when <contents> LEAVES
            # <container>. The sign is opposite to incontact/checkdistance -- the pair
            # must stay TOGETHER -- so it is deliberately kept out of `pairs`, whose
            # whole machinery (min distance, retreat direction) means "get away from".
            preserve_pairs.append((str(state[1]), str(state[2])))
        elif state[0] == "fall" and len(state) >= 2:
            fall_objects.append(str(state[1]))
    distances: list[float] = []
    contact_count = 0
    pair_names: list[str] = []
    best_pair: str | None = None
    best_target_position: list[float] | None = None
    best_hazard_position: list[float] | None = None
    best_delta: list[float] | None = None
    best_distance: float | None = None
    for object_a, object_b in pairs:
        pair_name = f"{object_a}:{object_b}"
        pair_names.append(pair_name)
        pos_a = _object_position(env, object_a)
        pos_b = _object_position(env, object_b)
        distance = _distance_between_positions(pos_a, pos_b)
        if distance is not None and math.isfinite(distance):
            distances.append(distance)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_pair = pair_name
                best_target_position = pos_a
                best_hazard_position = pos_b
                best_delta = [float(a) - float(b) for a, b in zip(pos_a, pos_b)]
        if _objects_in_contact(env, object_a, object_b):
            contact_count += 1
    fall_details = [_fall_features(env, object_name) for object_name in fall_objects]
    fall_true_names = [detail["object"] for detail in fall_details if detail.get("fall")]
    return {
        "cost_predicate_values": predicate_values,
        "cost_pair_count": len(pairs),
        "cost_pair_min_distance": min(distances) if distances else 0.0,
        "cost_pair_contacts": contact_count,
        "cost_pair_names": pair_names,
        "cost_pair_min_name": best_pair,
        "cost_pair_target_position": best_target_position or [],
        "cost_pair_hazard_position": best_hazard_position or [],
        "cost_pair_delta_xyz": best_delta or [],
        "cost_pair_delta_xy": (best_delta or [])[:2],
        "cost_pair_distance_threshold": (min(distance_thresholds)
                                         if distance_thresholds else None),
        "cost_predicate_kinds": sorted(set(predicate_kinds)),
        "cost_semantics": (
            "preserve" if "noton" in predicate_kinds
            else "distance" if "checkdistance" in predicate_kinds
            else "contact" if "incontact" in predicate_kinds
            else "unknown"),
        "preserve_pair_names": [f"{a}:{b}" for a, b in preserve_pairs],
        "preserve_pair_count": len(preserve_pairs),
        "fall_object_count": len(fall_objects),
        "fall_true_count": len(fall_true_names),
        "fall_names": fall_objects,
        "fall_true_names": fall_true_names,
        "fall_max_position_diff": max((float(detail.get("position_diff", 0.0) or 0.0) for detail in fall_details), default=0.0),
        "fall_max_xy_diff": max((float(detail.get("xy_diff", 0.0) or 0.0) for detail in fall_details), default=0.0),
        "fall_max_height_drop": max((float(detail.get("height_drop", 0.0) or 0.0) for detail in fall_details), default=0.0),
        "fall_max_angle_abs": max((float(detail.get("angle_abs", 0.0) or 0.0) for detail in fall_details), default=0.0),
        "fall_details": fall_details,
        "eef_position": _eef_position(env, obs),
    }


def unwrap_arena_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    for _ in range(8):
        if id(current) in seen:
            break
        seen.add(id(current))
        if hasattr(current, "parsed_problem") and hasattr(current, "sim"):
            return current
        if hasattr(current, "env"):
            current = getattr(current, "env")
            continue
        if hasattr(current, "unwrapped"):
            current = getattr(current, "unwrapped")
            continue
        break
    return current


def _object_distance(env: Any, object_a: str, object_b: str) -> float | None:
    return _distance_between_positions(_object_position(env, object_a), _object_position(env, object_b))


def _object_position(env: Any, object_name: str) -> list[float] | None:
    try:
        body_ids = getattr(env, "obj_body_id")
        position = getattr(env.sim.data, "body_xpos")[body_ids[object_name]]
        return [float(value) for value in position]
    except Exception:
        return None


def _eef_position(env: Any, obs: Any = None) -> list[float]:
    # Observation dict is the most reliable source: VLA-Arena/robosuite expose
    # robot0_eef_pos directly, independent of sim internals.
    if isinstance(obs, dict):
        for key in ("robot0_eef_pos", "eef_pos", "robot0_eef_position"):
            try:
                value = obs.get(key)
                if value is not None and len(list(value)) >= 3:
                    return [float(item) for item in list(value)[:3]]
            except Exception:
                pass
    for attr in ("_eef_xpos", "eef_pos", "eef_position"):
        try:
            value = getattr(env, attr)
            if callable(value):
                value = value()
            if value is not None:
                return [float(item) for item in list(value)[:3]]
        except Exception:
            pass
    try:
        robot = (getattr(env, "robots", []) or [None])[0]
        value = getattr(robot, "_eef_xpos", None)
        if value is not None:
            return [float(item) for item in list(value)[:3]]
    except Exception:
        pass
    # Multi-arm robosuite keys eef_site_id by arm name ({"right": id, ...}).
    try:
        robot = (getattr(env, "robots", []) or [None])[0]
        site_id = getattr(robot, "eef_site_id", None)
        if isinstance(site_id, dict) and site_id:
            site_id = site_id.get("right", next(iter(site_id.values())))
        if site_id is not None:
            return [float(item) for item in env.sim.data.site_xpos[int(site_id)][:3]]
    except Exception:
        pass
    try:
        sim = env.sim
        candidates = [
            "gripper0_right_grip_site",
            "gripper0_grip_site",
            "robot0_grip_site",
            "eef_site",
            "end_effector",
        ]
        try:
            names = list(getattr(sim.model, "site_names", []) or [])
            candidates += [name for name in names if name.endswith("grip_site") and name not in candidates]
        except Exception:
            pass
        for name in candidates:
            try:
                site_id = sim.model.site_name2id(name)
                return [float(item) for item in sim.data.site_xpos[site_id][:3]]
            except Exception:
                continue
    except Exception:
        pass
    return []


def _object_quat(env: Any, object_name: str) -> list[float] | None:
    try:
        body_ids = getattr(env, "obj_body_id")
        quat = getattr(env.sim.data, "body_xquat")[body_ids[object_name]]
        return [float(value) for value in quat]
    except Exception:
        return None


def _original_position(env: Any, object_name: str) -> list[float] | None:
    try:
        value = getattr(env, "object_original_pos", {}).get(object_name)
        if value is None:
            return None
        return [float(item) for item in value]
    except Exception:
        return None


def _original_quat(env: Any, object_name: str) -> list[float] | None:
    try:
        value = getattr(env, "object_original_quat", {}).get(object_name)
        if value is None:
            return None
        return [float(item) for item in value]
    except Exception:
        return None


def _fall_features(env: Any, object_name: str) -> dict[str, Any]:
    current_pos = _object_position(env, object_name)
    original_pos = _original_position(env, object_name)
    current_quat = _object_quat(env, object_name)
    original_quat = _original_quat(env, object_name)
    position_diff = _distance_between_positions(current_pos, original_pos) or 0.0
    xy_diff = _distance_between_positions((current_pos or [])[:2], (original_pos or [])[:2]) or 0.0
    height_drop = 0.0
    if current_pos is not None and original_pos is not None and len(current_pos) >= 3 and len(original_pos) >= 3:
        height_drop = float(original_pos[2]) - float(current_pos[2])
    angle_abs = _quat_angle_abs(current_quat, original_quat)
    state = ["fall", object_name]
    value = _cost_predicate_value(env, state).get("value")
    return {
        "object": object_name,
        "fall": bool(value) if value is not None else False,
        "position_diff": float(position_diff),
        "xy_diff": float(xy_diff),
        "height_drop": float(height_drop),
        "angle_abs": float(angle_abs),
        "current_position": current_pos or [],
        "original_position": original_pos or [],
    }


def _quat_angle_abs(current_quat: list[float] | None, original_quat: list[float] | None) -> float:
    if current_quat is None or original_quat is None or len(current_quat) != 4 or len(original_quat) != 4:
        return 0.0
    try:
        from robosuite.utils import transform_utils

        diff = transform_utils.quat_multiply(current_quat, transform_utils.quat_inverse(original_quat))
        axis_angle = transform_utils.quat2axisangle(diff)
        return max(abs(float(value)) for value in axis_angle)
    except Exception:
        return 0.0


def _cost_predicate_value(env: Any, state: Any) -> dict[str, Any]:
    label = ":".join(str(part) for part in state) if isinstance(state, (list, tuple)) else str(state)
    try:
        value = bool(env._eval_predicate(state))
        return {"predicate": label, "state": list(state), "value": value}
    except Exception as exc:
        return {"predicate": label, "state": list(state) if isinstance(state, (list, tuple)) else [state], "value": None, "error": type(exc).__name__}


def _distance_between_positions(pos_a: list[float] | None, pos_b: list[float] | None) -> float | None:
    if pos_a is None or pos_b is None:
        return None
    return sum((float(a) - float(b)) ** 2 for a, b in zip(pos_a, pos_b)) ** 0.5


def _objects_in_contact(env: Any, object_a: str, object_b: str) -> bool:
    try:
        sim = env.sim
        names_a = _object_geom_name_candidates(object_a)
        names_b = _object_geom_name_candidates(object_b)
        for idx in range(int(getattr(sim.data, "ncon", 0))):
            contact = sim.data.contact[idx]
            geom_1 = str(sim.model.geom_id2name(contact.geom1) or "")
            geom_2 = str(sim.model.geom_id2name(contact.geom2) or "")
            if (_matches_object_geom(geom_1, names_a) and _matches_object_geom(geom_2, names_b)) or (_matches_object_geom(geom_1, names_b) and _matches_object_geom(geom_2, names_a)):
                return True
    except Exception:
        return False
    return False


def _object_geom_name_candidates(object_name: str) -> list[str]:
    base = object_name.replace("_1", "")
    return [object_name, base]


def _matches_object_geom(geom_name: str, candidates: list[str]) -> bool:
    return any(candidate in geom_name for candidate in candidates)
