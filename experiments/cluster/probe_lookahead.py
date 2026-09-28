#!/usr/bin/env python3
"""Identity check + effect check for the hazard lookahead.

1. lookahead_s=0 must reproduce the original snapshot geometry EXACTLY. Everything measured
   so far (the 300-cell pi_2 vs AEGIS comparison, the whole static line) was produced with the
   old function; if the default drifted even slightly those results stop being comparable.
   The original is loaded from the .bak taken at patch time and run side by side.
2. lookahead_s>0 must move the sphere by v*tau and no more.
"""
import os, sys, glob, importlib.util
import numpy as np

sys.path.insert(0, "${SE_VLA_ROOT}")
sys.path.insert(0, "${SE_VLA_ROOT}/external/VLA-Arena")

from vla_arena.vla_arena import benchmark, get_vla_arena_path
from vla_arena.vla_arena.envs import OffScreenRenderEnv
from roundG.pi05_stage2.vlsa_aegis.geometry import (
    extract_oracle_ellipsoids, _body_world_velocity)

bak = sorted(glob.glob("${SE_VLA_ROOT}/roundG/pi05_stage2/vlsa_aegis/"
                       "geometry.py.bak.*"))[-1]
# spec_from_file_location returns None for a non-.py suffix, so stage a real module file
import shutil, tempfile
_tmp = tempfile.mkdtemp()
_staged = os.path.join(_tmp, "geometry_orig.py")
shutil.copy(bak, _staged)
sys.path.insert(0, _tmp)
spec = importlib.util.spec_from_file_location("geometry_orig", _staged)
assert spec is not None and spec.loader is not None, f"cannot load {_staged}"
orig = importlib.util.module_from_spec(spec)
sys.modules["geometry_orig"] = orig
spec.loader.exec_module(orig)
print(f"original loaded from {os.path.basename(bak)}")
assert not hasattr(orig, "_body_world_velocity"), "the .bak already contains the patch"

suite = benchmark.get_benchmark_dict()["safety_dynamic_obstacles"]()
task = suite.get_task_by_level_id(2, 0)
states = suite.get_task_init_states(2, 0)
bddl = os.path.join(get_vla_arena_path("bddl_files"), task.problem_folder,
                    f"level_{task.level}", task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
env.seed(17); env.reset(); env.set_init_state(states[0])
base = env.env if hasattr(env, "env") else env

zero = np.zeros(7)
identical = True
print()
print(f"{'step':>4}  {'orig center':>34}  {'new tau=0':>34}  {'一致':>5}")
for k in range(8):
    env.step(zero)
    a = orig.extract_oracle_ellipsoids(env, default_radius=0.04)
    b = extract_oracle_ellipsoids(env, default_radius=0.04, lookahead_s=0.0)
    assert len(a) == len(b) == 1, f"count mismatch {len(a)} vs {len(b)}"
    same = np.array_equal(a[0].center, b[0].center) and np.array_equal(a[0].radii, b[0].radii)
    identical &= same
    print(f"{k:>4}  {np.round(a[0].center,5)!s:>34}  {np.round(b[0].center,5)!s:>34}  {str(same):>5}")
print("IDENTITY", "PASS" if identical else "FAIL")
assert identical, "lookahead=0 changed the geometry -- every prior result is now incomparable"

print()
print("=== 环境变量默认路径也必须是 0 ===")
os.environ.pop("SE_VLA_HAZARD_LOOKAHEAD_S", None)
c_default = extract_oracle_ellipsoids(env, default_radius=0.04)[0].center
c_zero = extract_oracle_ellipsoids(env, default_radius=0.04, lookahead_s=0.0)[0].center
print(f"  未设环境变量: {np.round(c_default,5)}   显式0: {np.round(c_zero,5)}   一致={np.array_equal(c_default,c_zero)}")
assert np.array_equal(c_default, c_zero)

print()
print("=== lookahead 的实际效果 (hazard 速度 ~0.25 m/s, oracle_radius 0.04) ===")
bid = base.obj_body_id["mickey_1"]
v = _body_world_velocity(base, bid)
print(f"  hazard 速度 = {np.round(v,4)}  |v| = {np.linalg.norm(v):.4f} m/s")
c0 = extract_oracle_ellipsoids(env, default_radius=0.04, lookahead_s=0.0)[0].center
for tau in (0.1, 0.2, 0.4):
    c = extract_oracle_ellipsoids(env, default_radius=0.04, lookahead_s=tau)[0].center
    shift = np.linalg.norm(c - c0)
    print(f"  tau={tau:<4} 位移={shift:.4f} m  ({shift/0.04:.2f} 个 oracle_radius)  "
          f"与 v*tau 差={abs(shift - np.linalg.norm(v)*tau):.2e}")
    assert np.allclose(c, c0 + v * tau), "shift is not exactly v*tau"

print()
print("=== 环境变量通路 ===")
os.environ["SE_VLA_HAZARD_LOOKAHEAD_S"] = "0.2"
c_env = extract_oracle_ellipsoids(env, default_radius=0.04)[0].center
c_arg = extract_oracle_ellipsoids(env, default_radius=0.04, lookahead_s=0.2)[0].center
print(f"  env=0.2 -> {np.round(c_env,5)}   arg=0.2 -> {np.round(c_arg,5)}   一致={np.array_equal(c_env,c_arg)}")
assert np.array_equal(c_env, c_arg)
env.close()
print("LOOKAHEAD_PROBE_PASS")
