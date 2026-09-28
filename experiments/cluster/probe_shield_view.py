#!/usr/bin/env python3
"""Two questions before touching geometry.py:

  1. Does the shield actually see the moving hazard? The motion probe printed an empty
     hazard list, which would mean extract_oracle_ellipsoids returns nothing on L2t0 --
     a far worse bug than a stale snapshot, and one that would invalidate the dynamic
     conclusions entirely. Most likely my probe compared MuJoCo body names (mickey_1_main)
     against BDDL keys (mickey_1), but "most likely" is not verified.
  2. Which velocity accessor is correct? body_xvelp is gone; cvel exists but is a
     com-frame spatial velocity (angular first, then linear), not what we want directly.
     mj_objectVelocity with flg_local=0 gives world-frame. Whichever we use must agree
     with finite differences of body_xpos, so that is the ground truth here.
"""
import os, sys
import numpy as np

sys.path.insert(0, "${SE_VLA_ROOT}")
sys.path.insert(0, "${SE_VLA_ROOT}/external/VLA-Arena")

from vla_arena.vla_arena import benchmark, get_vla_arena_path
from vla_arena.vla_arena.envs import OffScreenRenderEnv
from roundG.pi05_stage2.vlsa_aegis.geometry import extract_oracle_ellipsoids

suite = benchmark.get_benchmark_dict()["safety_dynamic_obstacles"]()
task = suite.get_task_by_level_id(2, 0)
states = suite.get_task_init_states(2, 0)
bddl = os.path.join(get_vla_arena_path("bddl_files"), task.problem_folder,
                    f"level_{task.level}", task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
env.seed(17); env.reset(); env.set_init_state(states[0])
base = env.env if hasattr(env, "env") else env

print("=== 1. shield 到底看到什么 ===")
parsed = getattr(base, "parsed_problem", {}) or {}
print("cost_state      :", parsed.get("cost_state"))
print("obj_of_interest :", parsed.get("obj_of_interest"))
print("obj_body_id keys:", sorted((getattr(base, "obj_body_id", {}) or {}).keys()))
els = extract_oracle_ellipsoids(env, default_radius=0.04)
print(f"extract_oracle_ellipsoids -> {len(els)} 个:")
for e in els:
    print(f"    label={e.label!r} center={np.round(e.center,4)} radii={np.round(e.radii,4)}")
assert els, "SHIELD SEES NOTHING -- dynamic conclusions must be revisited"

print()
print("=== 2. 速度取法对不对（与有限差分比）===")
import mujoco
sim = base.sim
m, d = sim.model, sim.data
raw_m = getattr(m, "_model", m); raw_d = getattr(d, "_data", d)
label = els[0].label
bid = base.obj_body_id[label]
zero = np.zeros(7)
prev = np.array(sim.data.body_xpos[bid], dtype=float).copy()
dt = float(getattr(m, "opt", getattr(raw_m, "opt")).timestep) * int(
    getattr(base, "control_timestep", 1) / float(getattr(m, "opt", getattr(raw_m, "opt")).timestep)
) if hasattr(base, "control_timestep") else None
print(f"hazard={label!r} body_id={bid}  sim.timestep={getattr(raw_m,'opt').timestep}"
      f"  control_timestep={getattr(base,'control_timestep',None)}")
print(f"{'step':>4} {'有限差分速度(m/s)':>24} {'cvel[3:6]':>26} {'objectVelocity':>26}")
for k in range(6):
    env.step(zero)
    cur = np.array(sim.data.body_xpos[bid], dtype=float).copy()
    ct = float(getattr(base, "control_timestep", 0.05) or 0.05)
    fd = (cur - prev) / ct
    prev = cur
    cv = np.array(sim.data.cvel[bid][3:6], dtype=float)
    try:
        res = np.zeros(6)
        mujoco.mj_objectVelocity(raw_m, raw_d, mujoco.mjtObj.mjOBJ_BODY, bid, res, 0)
        ov = res[3:6]
    except Exception as e:
        ov = np.array([np.nan] * 3)
    print(f"{k:>4} {np.round(fd,4)!s:>24} {np.round(cv,4)!s:>26} {np.round(ov,4)!s:>26}")
env.close()
print("SHIELDVIEW_DONE")
