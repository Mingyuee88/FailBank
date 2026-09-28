#!/usr/bin/env python3
"""Can the LIBERO 'spatial' crop box hold the three unrun Safety suites' hazards?

Extends probe_crop.py (unchanged) to safety_cautious_grasp / safety_hazard_avoidance /
safety_state_preservation at level_2, all five tasks each. Same box, same measurement:
take every geom of every oracle hazard, and ask whether its axis-aligned extent is
contained in z in (0.92,1.50), x,y in (-0.3,0.3).

Why this is a prerequisite and not a nicety: _crop_suite_for maps only
safety_static_obstacles, so the AEGIS arm aborts on every other suite. Guessing a box
would produce an empty point cloud, a shield that never engages, and a "no effect"
reading that is an artefact of the crop rather than a property of the method -- which
is exactly the false-positive our motivation figure would be accused of manufacturing.
"""
import os, sys
import numpy as np
sys.path.insert(0, "${SE_VLA_ROOT}")
sys.path.insert(0, "${SE_VLA_ROOT}/external/VLA-Arena")
from vla_arena.vla_arena import benchmark, get_vla_arena_path
from vla_arena.vla_arena.envs import OffScreenRenderEnv

BOX = dict(z=(0.92, 1.50), x=(-0.3, 0.3), y=(-0.3, 0.3))
SUITES = ["safety_cautious_grasp", "safety_hazard_avoidance", "safety_state_preservation"]
REFERENCE = [("static L1t2 (mapped, control)", "safety_static_obstacles", 1, 2)]

CASES = REFERENCE + [(f"{s.replace('safety_','')} L2t{t}", s, 2, t)
                     for s in SUITES for t in range(5)]

summary = []
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
        from roundG.pi05_stage2.vlsa_aegis.geometry import extract_oracle_ellipsoids
        els = extract_oracle_ellipsoids(env, default_radius=0.04)
        print("=" * 78)
        print(f"{label}  ({task.bddl_file})")
        if not els:
            print("  (no oracle hazards found)")
            summary.append((label, "NO-HAZARD")); env.close(); continue
        allok = True
        for e in els:
            bid = base.obj_body_id[e.label]
            # geoms may hang off DESCENDANT bodies, not the named body itself: flat_stove_1
            # has 0 direct geoms and 23 across 4 children. Scanning only direct geoms made an
            # earlier version of this probe report "no geoms" and mis-grade the suite.
            desc = set()
            for jb in range(sim.model.nbody):
                p_, hops = jb, 0
                while p_ != 0 and hops < 12:
                    p_ = sim.model.body_parentid[p_]; hops += 1
                    if p_ == bid: desc.add(jb); break
            gids = [g for g in range(sim.model.ngeom)
                    if sim.model.geom_bodyid[g] in (desc | {bid})]
            if not gids:
                print(f"  {e.label}: no geoms (incl. descendants)"); allok = False; continue
            pos = np.array([sim.data.geom_xpos[g] for g in gids])
            sz = np.array([sim.model.geom_size[g] for g in gids])
            lo = (pos - sz[:, :3]).min(axis=0); hi = (pos + sz[:, :3]).max(axis=0)
            inx = BOX["x"][0] <= lo[0] and hi[0] <= BOX["x"][1]
            iny = BOX["y"][0] <= lo[1] and hi[1] <= BOX["y"][1]
            inz = BOX["z"][0] <= lo[2] and hi[2] <= BOX["z"][1]
            ok = inx and iny and inz
            allok = allok and ok
            print(f"  hazard={e.label}")
            print(f"    x [{lo[0]:+.3f},{hi[0]:+.3f}] {'OK' if inx else 'OUT'}   "
                  f"y [{lo[1]:+.3f},{hi[1]:+.3f}] {'OK' if iny else 'OUT'}   "
                  f"z [{lo[2]:+.3f},{hi[2]:+.3f}] {'OK' if inz else 'OUT'}")
            print(f"    -> {'inside the box' if ok else 'WOULD BE CROPPED AWAY'}")
        summary.append((label, "FITS" if allok else "CROPPED"))
        env.close()
    except Exception as e:
        print(f"{label}: FAILED {type(e).__name__}: {e}")
        summary.append((label, f"FAILED {type(e).__name__}"))

print("=" * 78)
print("SUMMARY (a suite may only be added to _crop_suite_for if EVERY task says FITS)")
for label, verdict in summary:
    print(f"  {label:34s} {verdict}")
print("CROP_PROBE_NEWSUITES_DONE")
