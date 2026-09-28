#!/usr/bin/env python3
"""Measure real hazard geometry against the sphere the shield actually uses.

`extract_oracle_ellipsoids` represents EVERY hazard as `Ellipsoid.sphere(body_xpos,
default_radius)` -- one fixed radius (0.04 in our runs), centred on the body origin --
even though `Ellipsoid` carries three radii and a rotation.

Two observations point at that approximation:
  * level_2 onion: pi_2 alone scores SR 41.3, but with the shield in the loop it drops to
    0.0. Shrinking clearance from ~0.09 to ~0.05 recovered only SR 4.0. If the target itself
    sits inside the clearance sphere, the barrier can never allow the approach.
  * L1t3: contact distances overlap between touching (0.0662-0.2001) and not touching
    (0.1745-0.1955) -- the signature of a non-spherical hazard (white_storage_box is a box).

So: per hazard, the true geom half-extents, and the distance from each task object to the
hazard versus the clearance the shield enforces. If clearance exceeds that distance, the
task is geometrically unreachable under the shield -- a property of the approximation, not
of safety.
"""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, "${SE_VLA_ROOT}/external/VLA-Arena")
import numpy as np

TARGETS = [
    ("level_1", "pick_the_onion_and_place_it_on_the_bowl_1.bddl", "L1t3 onion+storage_box"),
    ("level_2", "pick_the_onion_and_place_it_on_the_bowl_2.bddl", "L2 onion+2 wine bottles"),
    ("level_2", "pick_the_mango_and_place_it_on_the_bowl_2.bddl", "L2 mango+2 red mugs"),
]
EEF_R, MARGIN, ORACLE_R = 0.03, 0.02, 0.04

GEOM_TYPE = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid",
             5: "cylinder", 6: "box", 7: "mesh"}


def main():
    from vla_arena.vla_arena.envs import OffScreenRenderEnv
    from vla_arena.vla_arena import get_vla_arena_path
    root = get_vla_arena_path("bddl_files")
    for level, bddl, label in TARGETS:
        path = os.path.join(root, "safety_static_obstacles", level, bddl)
        if not os.path.isfile(path):
            print("MISSING %s" % path); continue
        env = OffScreenRenderEnv(bddl_file_name=path, camera_heights=128, camera_widths=128)
        env.reset()
        base = env.env if hasattr(env, "env") else env
        for _ in range(6):
            if getattr(base, "sim", None) is not None and isinstance(getattr(base, "obj_body_id", None), dict):
                break
            base = getattr(base, "env", None)
        parsed = getattr(base, "parsed_problem", {}) or {}
        def flat(v):
            if isinstance(v, str): yield v
            elif isinstance(v, dict):
                for x in v.values(): yield from flat(x)
            elif isinstance(v, (list, tuple, set)):
                for x in v: yield from flat(x)
        cost = set(flat(parsed.get("cost_state", [])))
        task = list(parsed.get("obj_of_interest", []))
        names = set(base.obj_body_id)
        hazards = sorted((cost & names) - set(task))
        model, data = base.sim.model, base.sim.data
        print("\n=== %s ===" % label)
        print("  task objects: %s" % task)
        print("  hazards     : %s" % hazards)
        for h in hazards:
            bid = base.obj_body_id[h]
            centre = np.asarray(data.body_xpos[bid], dtype=float)
            ext = np.zeros(3); kinds = []
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] != bid: continue
                gs = np.asarray(model.geom_size[gid], dtype=float)
                gt = int(model.geom_type[gid]); kinds.append(GEOM_TYPE.get(gt, str(gt)))
                gpos = np.asarray(model.geom_pos[gid], dtype=float)
                if gt == 2:   half = np.full(3, gs[0])
                elif gt == 6: half = gs[:3]
                elif gt in (3, 5): half = np.array([gs[0], gs[0], gs[1] + (gs[0] if gt == 3 else 0)])
                elif gt == 4: half = gs[:3]
                else:         half = np.full(3, float(np.max(gs)) if np.any(gs) else 0.0)
                ext = np.maximum(ext, np.abs(gpos) + half)
            print("    %-26s geoms=%s" % (h, sorted(set(kinds))))
            print("      true half-extents  x %.4f  y %.4f  z %.4f   (max %.4f)"
                  % (ext[0], ext[1], ext[2], ext.max()))
            print("      shield sphere radius %.4f  -> %s by %.4f on the smallest axis"
                  % (ORACLE_R, "OVER" if ORACLE_R > ext.min() else "under",
                     abs(ORACLE_R - ext.min())))
            clearance = EEF_R + ORACLE_R + MARGIN
            for t in task:
                if t not in base.obj_body_id: continue
                tp = np.asarray(data.body_xpos[base.obj_body_id[t]], dtype=float)
                d = float(np.linalg.norm(tp - centre))
                verdict = ("UNREACHABLE: target inside clearance" if d < clearance
                           else "reachable, %.4f of room" % (d - clearance))
                print("      dist to %-22s %.4f  vs clearance %.4f  -> %s" % (t, d, clearance, verdict))
        try: env.close()
        except Exception: pass


if __name__ == "__main__":
    main()
