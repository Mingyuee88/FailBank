#!/usr/bin/env python3
"""Seam 2 of the E3 port: which workspace range does filtering_points need here?

`filtering_points(pts, task_suite_name)` crops the point cloud to a hard-coded XYZ box
chosen by suite name:

    spatial / goal : z in (0.92, 1.50)   x,y in (-0.3, 0.3)   table scene
    object         : z in (0.05, 0.50)   x,y in (-0.3, 0.3)   floor scene
    long           : z in (0.43, 0.80)   x,y in (-0.3, 0.3)   living-room table

VLA-Arena's suite names ("safety_static_obstacles", "safety_hazard_avoidance", ...) match
none of those, so `keep` is never assigned and the function raises UnboundLocalError. The
port has to choose a box, and the choice must come from measured geometry rather than from
the fact that both benchmarks say "tabletop".

The box exists to drop the table surface and keep what stands on it. So this measures, on
a real reset of each target arena: the table top height, and the z-extent of the objects
the cost predicate names. A correct box sits just above the table and contains the hazard.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, "${SE_VLA_ROOT}/external/VLA-Arena")

import numpy as np

TARGETS = [
    ("safety_static_obstacles", "level_1", "pick_the_mango_and_place_it_on_the_bowl_1.bddl"),
    ("safety_hazard_avoidance", "level_1",
     "pick_up_the_onion_and_place_it_on_the_akita_black_bowl_with_the_stove_turned_on.bddl"),
]

# the candidate boxes, verbatim from filtering_points
BOXES = {
    "spatial/goal (table)": ((0.92, 1.50), (-0.3, 0.3), (-0.3, 0.3)),
    "object (floor)":       ((0.05, 0.50), (-0.3, 0.3), (-0.3, 0.3)),
    "long (living room)":   ((0.43, 0.80), (-0.3, 0.3), (-0.3, 0.3)),
}


def main():
    from vla_arena.vla_arena.envs import OffScreenRenderEnv
    from vla_arena.vla_arena import get_vla_arena_path

    for suite, level, bddl_name in TARGETS:
        bddl = os.path.join(get_vla_arena_path("bddl_files"), suite, level, bddl_name)
        if not os.path.exists(bddl):
            print(f"MISSING bddl {bddl}")
            continue
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_names=["agentview", "robot0_eye_in_hand", "backview"],
            camera_heights=256, camera_widths=256, camera_depths=True,
        )
        env.seed(0)
        env.reset()
        base = env.env
        while hasattr(base, "env") and not hasattr(base, "obj_body_id"):
            base = base.env

        parsed = getattr(base, "parsed_problem", {}) or {}
        cost_state = parsed.get("cost_state", []) or []
        interest = set(parsed.get("obj_of_interest", []))

        def flat(v):
            if isinstance(v, str):
                yield v
            elif isinstance(v, (list, tuple)):
                for x in v:
                    yield from flat(x)
        named = sorted(set(flat(cost_state)) & set(base.obj_body_id))

        print(f"\n=== {suite} / {bddl_name}")
        print(f"    scene xml: {getattr(base, '_arena_xml', '?')}")
        print(f"    cost_state objects: {named}   obj_of_interest: {sorted(interest)}")

        # table surface: use the workspace body if present, else the lowest cost object
        zs = {}
        for name in named:
            try:
                pos = np.asarray(base.sim.data.body_xpos[base.obj_body_id[name]], dtype=float)
            except Exception:
                continue
            zs[name] = pos
            tag = "TARGET" if name in interest else "hazard"
            print(f"    {tag:7s} {name:28s} pos {np.round(pos, 4)}")

        # geom-level z extent of the hazard, which is what the point cloud actually sees
        for name, pos in zs.items():
            if name in interest:
                continue
            gids = [i for i in range(base.sim.model.ngeom)
                    if base.sim.model.geom_bodyid[i] == base.obj_body_id[name]]
            if gids:
                zlo = min(base.sim.data.geom_xpos[i][2] - base.sim.model.geom_rbound[i] for i in gids)
                zhi = max(base.sim.data.geom_xpos[i][2] + base.sim.model.geom_rbound[i] for i in gids)
                print(f"    hazard {name} geom z-extent approx [{zlo:.4f}, {zhi:.4f}]")
                for label, (zr, xr, yr) in BOXES.items():
                    inside = (zr[0] < pos[2] < zr[1]) and (xr[0] < pos[0] < xr[1]) and (yr[0] < pos[1] < yr[1])
                    overlap = not (zhi < zr[0] or zlo > zr[1])
                    print(f"      box {label:22s} centre-inside={inside!s:5s} z-overlap={overlap}")

        env.close()

    print("\nSEAM2_GEOMETRY_PROBED")


if __name__ == "__main__":
    main()
