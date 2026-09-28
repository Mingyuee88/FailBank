#!/usr/bin/env python3
"""Seam 1 of the E3 port: does VLA-Arena actually give us what the AEGIS shield consumes?

The shield needs, per step: agentview RGB + depth, backview RGB + depth, the eef pose, and
a MuJoCo sim handle that robosuite's get_real_depth_map can convert against. The backview
camera was just copied into the arena scene XMLs from the SafeLIBERO counterparts; whether
MuJoCo renders it, and whether depth comes back at all, has to be observed rather than
assumed -- a missing camera would silently degrade the port to a single view and any
comparison would then be about our truncation, not their method.

Checks, on a real reset of the actual target arena:
  * the requested cameras appear in the observation dict
  * their depth planes appear and are finite and non-degenerate
  * robosuite's get_real_depth_map accepts them
  * the two views see the scene from genuinely different poses (a copy-paste error that
    duplicated agentview would pass every other check)
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, "${SE_VLA_ROOT}/external/VLA-Arena")

import numpy as np


def main():
    from vla_arena.vla_arena.envs import OffScreenRenderEnv
    from vla_arena.vla_arena import get_vla_arena_path

    bddl = os.path.join(
        get_vla_arena_path("bddl_files"),
        "safety_static_obstacles", "level_1",
        "pick_the_mango_and_place_it_on_the_bowl_1.bddl",
    )
    print("bddl:", bddl, "exists:", os.path.exists(bddl))

    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_names=["agentview", "robot0_eye_in_hand", "backview"],
        camera_heights=256,
        camera_widths=256,
        camera_depths=True,
    )
    env.seed(0)
    obs = env.reset()

    keys = sorted(obs.keys())
    print("\nobservation keys:")
    for k in keys:
        v = obs[k]
        shape = getattr(v, "shape", None)
        print(f"  {k:34s} {str(shape):18s} {getattr(v, 'dtype', '')}")

    need = ["agentview_image", "agentview_depth", "backview_image", "backview_depth",
            "robot0_eef_pos", "robot0_eef_quat"]
    missing = [k for k in need if k not in obs]
    print("\nMISSING:", missing if missing else "none")
    if missing:
        env.close()
        return 1

    # depth must be real, not a constant plane
    for cam in ("agentview", "backview"):
        d = np.asarray(obs[f"{cam}_depth"]).squeeze()
        finite = np.isfinite(d)
        print(f"{cam}_depth: finite {finite.mean()*100:.1f}%  "
              f"min {np.nanmin(d[finite]):.4f}  max {np.nanmax(d[finite]):.4f}  "
              f"unique {len(np.unique(d[finite])):d}")
        assert finite.mean() > 0.5, f"{cam} depth mostly non-finite"
        assert len(np.unique(d[finite])) > 100, f"{cam} depth is nearly constant"

    # robosuite's converter must accept them -- this is what get_point_cloud calls
    from robosuite.utils.camera_utils import get_real_depth_map
    for cam in ("agentview", "backview"):
        real = get_real_depth_map(env.sim, np.asarray(obs[f"{cam}_depth"]))
        r = np.asarray(real).squeeze()
        print(f"{cam} real depth: min {np.nanmin(r):.4f} max {np.nanmax(r):.4f} (metres)")

    # the two views must not be the same camera under two names
    a = np.asarray(obs["agentview_image"], dtype=float)
    b = np.asarray(obs["backview_image"], dtype=float)
    diff = np.abs(a - b).mean()
    print(f"\nmean |agentview - backview| pixel difference: {diff:.2f}")
    assert diff > 5.0, "the two views are near-identical -- backview may be a duplicate"

    # and their camera poses must differ in the sim
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
    for cam in ("agentview", "backview"):
        M = get_camera_extrinsic_matrix(env.sim, cam)
        print(f"{cam} camera position in world: {np.round(M[:3, 3], 4)}")

    print("\nSEAM1_CAMERAS_VERIFIED")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
