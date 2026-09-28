#!/usr/bin/env python3
"""Does the LIBERO 'spatial' crop box actually contain the dynamic suites' hazards?

That box is z in (0.92, 1.50), x,y in (-0.3, 0.3). _crop_suite_for refuses to map any suite
whose fit was never measured, and it is right to: on safety_hazard_avoidance the same box
erases the hazard entirely, which would look like "the shield has no effect" when it is
really "the shield sees nothing". Measure geom extents before mapping anything.
"""
import os, sys
import numpy as np
sys.path.insert(0, "${SE_VLA_ROOT}")
sys.path.insert(0, "${SE_VLA_ROOT}/external/VLA-Arena")
from vla_arena.vla_arena import benchmark, get_vla_arena_path
from vla_arena.vla_arena.envs import OffScreenRenderEnv

BOX = dict(z=(0.92, 1.50), x=(-0.3, 0.3), y=(-0.3, 0.3))
CASES = [("dyn L1t3", "safety_dynamic_obstacles", 1, 3),
         ("dyn L2t0", "safety_dynamic_obstacles", 2, 0),
         ("static L1t2", "safety_static_obstacles", 1, 2)]

for label, suite_name, lv, tid in CASES:
    try:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        task = suite.get_task_by_level_id(lv, tid)
        states = suite.get_task_init_states(lv, tid)
        bddl = os.path.join(get_vla_arena_path("bddl_files"), task.problem_folder,
                            f"level_{task.level}", task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        env.seed(17); env.reset(); env.set_init_state(states[0])
        base = env.env if hasattr(env, "env") else env
        sim = base.sim
        parsed = getattr(base, "parsed_problem", {}) or {}
        from roundG.pi05_stage2.vlsa_aegis.geometry import extract_oracle_ellipsoids
        els = extract_oracle_ellipsoids(env, default_radius=0.04)
        print("=" * 74)
        print(f"{label}  ({task.bddl_file})")
        for e in els:
            bid = base.obj_body_id[e.label]
            gids = [g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == bid]
            if not gids:
                print(f"  {e.label}: no geoms"); continue
            pos = np.array([sim.data.geom_xpos[g] for g in gids])
            sz = np.array([sim.model.geom_size[g] for g in gids])
            lo = (pos - sz[:, :3]).min(axis=0); hi = (pos + sz[:, :3]).max(axis=0)
            inx = BOX["x"][0] <= lo[0] and hi[0] <= BOX["x"][1]
            iny = BOX["y"][0] <= lo[1] and hi[1] <= BOX["y"][1]
            inz = BOX["z"][0] <= lo[2] and hi[2] <= BOX["z"][1]
            print(f"  hazard={e.label}")
            print(f"    x [{lo[0]:+.3f},{hi[0]:+.3f}] {'OK' if inx else 'OUT'}   "
                  f"y [{lo[1]:+.3f},{hi[1]:+.3f}] {'OK' if iny else 'OUT'}   "
                  f"z [{lo[2]:+.3f},{hi[2]:+.3f}] {'OK' if inz else 'OUT'}")
            print(f"    -> {'完全落在 crop box 内' if (inx and iny and inz) else '会被 crop box 裁掉'}")
        env.close()
    except Exception as e:
        print(f"{label}: FAILED {type(e).__name__}: {e}")
print("CROP_PROBE_DONE")
