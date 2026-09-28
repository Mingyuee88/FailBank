"""Reproduce the two-obstacle deadlock in isolation, and check the joint solver fixes it.

Geometry mirrors level_2 onion: the gripper sits between two wine bottles and wants to move
straight ahead (+y). Each bottle is off to one side (+/-x), so neither actually blocks +y --
a correct shield should leave the y-motion essentially alone.
"""
import sys, os, importlib
sys.path.insert(0, "${SE_VLA_ROOT}/roundG/pi05_stage2")
import numpy as np
from vlsa_aegis.geometry import Ellipsoid

obs = [Ellipsoid.sphere([0.09, 0.0, 0.0], 0.05, "bottle_1"),
       Ellipsoid.sphere([-0.09, 0.0, 0.0], 0.05, "bottle_2")]
kw = dict(alpha=3.0, max_translation=1.0, eef_radius=0.03, obstacle_margin=0.02)
NOM = [0.0, 1.0, 0.0]          # straight ahead, between the two bottles
EEF = [0.0, 0.0, 0.0]          # exactly in the middle

def run(iters):
    os.environ["SE_VLA_CBF_JOINT_ITERS"] = str(iters)
    import vlsa_aegis.cbf_qp as q
    importlib.reload(q)
    return q.project_translation(NOM, EEF, obs, **kw)

print("nominal velocity: %s   (pure +y, neither bottle is in the way)" % NOM)
print("each bottle: centre +/-0.09 on x, radius 0.05; clearance = 0.03+0.05+0.02 = 0.10")
print("so barrier = 0.09 - 0.10 = -0.01 on BOTH -> both demand a large speed, in opposite directions\n")
for iters in (0, 1, 4, 12, 40):
    r = run(iters)
    v = np.round(r.translation, 4)
    ymag = abs(float(r.translation[1]))
    print("  JOINT_ITERS=%-3d status=%-18s v=%s   |v_y|=%.4f  %s"
          % (iters, r.status, v, ymag,
             "FORWARD MOTION LOST" if ymag < 0.5 else "forward motion preserved"))
print("\nJOINT_ITERS=0 is the shipped sequential routine (and the published AEGIS path).")
