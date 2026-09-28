#!/usr/bin/env python3
"""Seam 2b: is 0.92 a frame constant or a LIBERO-workspace constant?

filtering_points crops the table scene to z > 0.92. Its purpose is to strip the table
surface and keep what stands on it. Whether that number transfers to VLA-Arena depends on
whether the two benchmarks put their table top at the same height -- and whether the
arena's hazards are the same KIND of object.

The `safety_static_obstacles` hazard is a mug whose geoms span z 0.909 to 1.134, so a
0.92 floor keeps almost all of it. The `safety_hazard_avoidance` hazard is a flat stove
with its body origin at z 0.905, which the same floor may erase entirely -- and an erased
hazard means an empty cloud, which means the shield silently never engages and the port
reports "no effect" for a reason that has nothing to do with the method.

This measures the table top height in both benchmarks and the true geom extent of each
hazard, so the port's choice of crop can be justified from geometry.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, "${SE_VLA_ROOT}/external/VLA-Arena")

import numpy as np


def geom_extent(base, body_name):
    """True axis-aligned z extent of every geom on a body, from the mesh bounds."""
    bid = base.obj_body_id.get(body_name)
    if bid is None:
        return None
    lo, hi, n = np.inf, -np.inf, 0
    for i in range(base.sim.model.ngeom):
        if base.sim.model.geom_bodyid[i] != bid:
            continue
        n += 1
        c = base.sim.data.geom_xpos[i]
        # geom_rbound is the bounding-sphere radius: conservative but honest
        r = base.sim.model.geom_rbound[i]
        lo = min(lo, c[2] - r)
        hi = max(hi, c[2] + r)
    return (lo, hi, n) if n else None


def table_top(base):
    """Highest z of any geom belonging to a body whose name looks like the workspace."""
    best = None
    for i in range(base.sim.model.ngeom):
        gname = base.sim.model.geom_id2name(i) or ""
        bid = base.sim.model.geom_bodyid[i]
        bname = base.sim.model.body_id2name(bid) or ""
        if not any(k in (gname + bname).lower() for k in ("table", "floor", "counter")):
            continue
        z = base.sim.data.geom_xpos[i][2] + base.sim.model.geom_size[i][2] \
            if base.sim.model.geom_type[i] == 6 else base.sim.data.geom_xpos[i][2]
        if best is None or z > best[0]:
            best = (z, bname, gname)
    return best


def probe(env_maker, label, hazards):
    env = env_maker()
    env.seed(0)
    env.reset()
    base = env.env
    while hasattr(base, "env") and not hasattr(base, "obj_body_id"):
        base = base.env
    print(f"\n=== {label}")
    t = table_top(base)
    print(f"    workspace top surface: {t}")
    for h in hazards:
        e = geom_extent(base, h)
        print(f"    hazard {h:28s} geom z extent {e}")
    env.close()


def main():
    from vla_arena.vla_arena.envs import OffScreenRenderEnv
    from vla_arena.vla_arena import get_vla_arena_path

    for suite, bddl_name, hazards in [
        ("safety_static_obstacles", "pick_the_mango_and_place_it_on_the_bowl_1.bddl",
         ["white_yellow_mug_1"]),
        ("safety_hazard_avoidance",
         "pick_up_the_onion_and_place_it_on_the_akita_black_bowl_with_the_stove_turned_on.bddl",
         ["flat_stove_1"]),
    ]:
        bddl = os.path.join(get_vla_arena_path("bddl_files"), suite, "level_1", bddl_name)
        probe(lambda b=bddl: OffScreenRenderEnv(
            bddl_file_name=b, camera_names=["agentview"], camera_heights=128,
            camera_widths=128, camera_depths=True), f"ARENA {suite}", hazards)

    print("\nSEAM2B_DONE")


if __name__ == "__main__":
    main()
