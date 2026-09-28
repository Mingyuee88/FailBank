#!/usr/bin/env python3
"""How far do hazard bodies actually move, dynamic suite vs static suite?

Claim under test: extract_oracle_ellipsoids turns each hazard into a fixed-radius sphere at
data.body_xpos -- where it is NOW. For a static obstacle "now" is also "when the arm arrives".
For a moving one it is not, so the barrier normal points at a stale position and the teacher's
correction direction is systematically wrong.

Two things measured here:
  1. Actual displacement of every non-robot body, against the shield's oracle_radius (0.04 m).
     If dynamic hazards move far less than that, the mechanism claim is WRONG and the
     extrapolation fix would be pointless.
  2. Whether a per-body linear velocity is even reachable from this MuJoCo build -- the
     proposed fix needs one. mujoco-py exposed data.body_xvelp; the current bindings may not,
     in which case finite differences of body_xpos are the fallback.
"""
import os, sys
import numpy as np

sys.path.insert(0, "${SE_VLA_ROOT}")
sys.path.insert(0, "${SE_VLA_ROOT}/external/VLA-Arena")

from vla_arena.vla_arena import benchmark, get_vla_arena_path
from vla_arena.vla_arena.envs import OffScreenRenderEnv

STEPS = 200
ORACLE_RADIUS = 0.04
CASES = [
    ("dynamic L2t0", "safety_dynamic_obstacles", 2, 0),
    ("dynamic L1t3", "safety_dynamic_obstacles", 1, 3),
    ("static  L1t2", "safety_static_obstacles", 1, 2),
]

vel_checked = False
for label, suite_name, level, tid in CASES:
    try:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        task = suite.get_task_by_level_id(level, tid)
        init_states = suite.get_task_init_states(level, tid)
        bddl = os.path.join(get_vla_arena_path("bddl_files"), task.problem_folder,
                            f"level_{task.level}", task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        env.seed(17)
        env.reset()
        env.set_init_state(init_states[0])

        base = env.env if hasattr(env, "env") else env
        sim = base.sim
        # the shield only ever looks at these -- BDDL cost-state objects minus task objects
        parsed = getattr(base, "parsed_problem", {}) or {}
        obj_ids = getattr(base, "obj_body_id", {}) or {}

        if not vel_checked:
            vel_checked = True
            d = sim.data
            print("--- velocity signal availability ---")
            for attr in ("body_xvelp", "body_xvelr", "cvel", "qvel"):
                has = hasattr(d, attr)
                shape = getattr(getattr(d, attr, None), "shape", None)
                print(f"    data.{attr:<12} present={has} shape={shape}")
            print()

        names = [sim.model.body_id2name(i) for i in range(sim.model.nbody)]
        keep = [n for n in names if n and not n.startswith(
            ("robot0", "gripper0", "world", "table", "mount"))]
        ids = {n: sim.model.body_name2id(n) for n in keep}
        traj = {n: [] for n in keep}
        zero = np.zeros(7)
        for _ in range(STEPS):
            base.sim.step() if not hasattr(env, "step") else env.step(zero)
            for n in keep:
                traj[n].append(np.array(sim.data.body_xpos[ids[n]]))

        print("=" * 82)
        print(f"{label}  ({suite_name} L{level} t{tid})  {task.bddl_file}")
        print(f"{STEPS} steps, arm held still, shield oracle_radius={ORACLE_RADIUS}")
        print(f"shield 会看的 hazard: {sorted(set(obj_ids) & set(keep))[:6]}")
        print(f"{'body':<30} {'总位移':>9} {'单步最大':>10} {'10步窗口':>10}")
        rows = []
        for n, xs in traj.items():
            xs = np.stack(xs)
            total = float(np.linalg.norm(xs[-1] - xs[0]))
            per = float(np.max(np.linalg.norm(np.diff(xs, axis=0), axis=1)))
            w = max(float(np.linalg.norm(xs[min(i + 10, len(xs) - 1)] - xs[i]))
                    for i in range(0, len(xs), 5))
            rows.append((w, total, per, n))
        for w, total, per, n in sorted(rows, reverse=True)[:7]:
            flag = "  <-- 超过 oracle_radius" if w > ORACLE_RADIUS else ""
            print(f"{n:<30} {total:9.4f} {per:10.5f} {w:10.4f}{flag}")
        env.close()
    except Exception as e:
        print(f"{label}: FAILED {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
