"""Published AEGIS (vlsa-aegis), running inside the VLA-Arena harness.

The shield's own functions are imported from the cloned repository and are NOT
reimplemented here: obstacle_detection (GLM-4.5v), get_point_cloud (GroundingDINO +
RGB-D), filtering_points, fit_ellipse and compute_h_coeffs_3d all come from
`vlsa-aegis/main/utils.py` exactly as published. What this module supplies is the glue
their `main_aegis_translational.py` / `main_aegis.py` provide around them -- the
per-episode perception pass, the per-step QP, and the constants -- rebuilt against
VLA-Arena's env instead of LIBERO's, with every constant copied from their file and
cited by line.

Why this shape. Hooking the arena's existing step seam rather than writing a new evaluator
means the policy server, the offsets, the cost accounting and the result.json are byte-for
-byte the same machinery every other arm in this study ran through. The only thing that
differs between "base", "our controller" and "published AEGIS" is which function computes
the executed action, which is what makes the three columns comparable at all.

TWO VARIANTS, selected by AEGIS_DOF (default "3", the variant the paper reports):

    "3"  main_aegis_translational.py -- their `Ours_t` column. Builds
             action_input = zeros(7); action_input[:3] = 0.2*R1@u_v; action_input[6] = gripper
         so the ROTATION CHANNEL IS DISCARDED AT EVERY STEP, active constraint or not.
         On translation it is the identity when the constraint is inactive
         (0.2 * R1 @ (5 * R1.T @ a[:3]) == a[:3]); on rotation it is identically zero.
    "6"  main_aegis.py -- the full formulation. u is 9-dimensional, omega enters both the
         objective and the constraint, and action_input[3:6] = 0.2*u_omega, which is the
         identity when inactive (0.2 * 5 * a[3:6] == a[3:6]).

Why the "6" variant exists here. On VLA-Arena the translational variant destroys 9 of 36
base successes while repairing 8 of 11 base failures. Those two sets are exactly
{base succeeded} and {base failed}, so "the shield destroys episodes that had no cost to
remove" and "the shield destroys episodes whose task needs wrist rotation" predict the
identical partition and cannot be separated observationally. Running the 6-DoF variant on
the same offsets is the intervention that separates them: if the destruction survives the
restoration of the rotation channel it is a property of the barrier, and if it does not it
was an artefact of the translational variant's amputation.

Constants, from main_aegis_translational.py / main_aegis.py:
    offset_local = [0, 0, -0.08]        barrier centre is 8 cm below the eef frame origin
    Q1_diag      = [0.06, 0.12, 0.2] or [0.06, 0.12, 0.11]   gripper ellipsoid, task-dependent
    Kp_pos       = 1
    u_v_ref      = 5 * v_ref            u_omega_ref = 5 * omega_ref   (6-DoF only)
    a_u_v        = 0.2 * a_v            a_u_omega   = 0.2 * a_omega   (6-DoF only)
    u_z_nom      = 10 * mu_row
    W            = diag([1/25]*3 + [1,1,1])          (translational)
                   diag([1/25]*6 + [1,1,1])          (6-DoF)
    constraint     a_u_v . u_v + [a_u_omega . u_omega] + a_uz . u_z + 10 * h >= 0
    executed       action[:3] = 0.2 * R1 @ u_v   [+ action[3:6] = 0.2 * u_omega]
    dt           = 0.05                 for the z_fixed update
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np

def _aegis_root() -> pathlib.Path:
    root = os.environ.get("AEGIS_ROOT", "")
    if not root:
        raise RuntimeError("AEGIS_ROOT must point at a vlsa-aegis checkout (see extras/aegis_baseline/README.md)")
    return pathlib.Path(root)


_AEGIS_MAIN = str(_aegis_root() / "main")
if _AEGIS_MAIN not in sys.path:
    sys.path.insert(0, _AEGIS_MAIN)

# their code, unmodified
from utils import (  # noqa: E402
    compute_h_coeffs_3d,
    filtering_points,
    fit_ellipse,
    get_point_cloud,
    obstacle_detection,
)

# main_aegis_translational.py line 168: the gripper ellipsoid is stretched in z for a few
# tall objects and shorter otherwise. Their test is on the task description string.
_TALL = ("orange juice", "milk", "alphabet soup")


def _quat_to_R(quat_xyzw):
    from scipy.spatial.transform import Rotation as R
    return R.from_quat(np.asarray(quat_xyzw, dtype=float)).as_matrix()


def _dof():
    d = os.environ.get("AEGIS_DOF", "3")
    if d not in ("3", "6"):
        raise ValueError(f"AEGIS_DOF must be 3|6, got {d!r}")
    return int(d)


class VlsaAegisShield:
    """One episode's worth of published-AEGIS state.

    Perception runs ONCE per episode, exactly as in their loop: they call
    obstacle_detection and get_point_cloud before the control while-loop and reuse the
    fitted ellipsoid for every step. The QP then runs per step.
    """

    def __init__(self, env, task_description, suite_name, out_dir, dino_model, episode_label=""):
        self.env = env
        self.episode_label = str(episode_label)
        self.task_description = str(task_description)
        self.suite_name = str(suite_name)
        # their get_point_cloud does `save_path / name`, so this must be a Path, not a str
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.dino = dino_model
        self.active = False
        self.z_fixed = None
        self.p2 = self.R2 = self.Q2_diag = None
        self.dof = _dof()
        self.audit = {
            "obstacle_name": None,
            "points_agentview": 0,
            "points_backview": 0,
            "points_fused": 0,
            "points_filtered": 0,
            "perception_ok": False,
            "steps": 0,
            "qp_solved": 0,
            "qp_infeasible": 0,
            "h_min": None,
            "override_norm_sum": 0.0,
            # new, and additive: nothing above changes meaning
            "dof": self.dof,
            "override_rot_norm_sum": 0.0,
            "constraint_active_steps": 0,
            "release_step": int(os.environ.get("AEGIS_RELEASE_STEP", "0") or 0),
            "released_steps": 0,
        }
        # Per-step state for the SRD recorder. The audit above is cumulative and cannot
        # answer "what did the shield do at THIS step", which is exactly what a teacher
        # record needs.
        self.last = {"triggered": False, "norm": 0.0, "h": None, "active_constraints": 0}
        self.Q1_diag = (np.array([0.06, 0.12, 0.2]) if any(t in self.task_description for t in _TALL)
                        else np.array([0.06, 0.12, 0.11]))
        # Per-step trace. The episode-level audit sums are not enough to build or even to
        # evaluate an intervention gate: a gate has to decide from what is knowable at the
        # step, and episode length is an OUTCOME, so any feature derived from it is
        # post-treatment. Everything written here is available at the moment of the
        # decision.
        self._steps_path = os.environ.get(
            "AEGIS_STEPS_PATH", str(self.out_dir / "vlsa_steps.jsonl"))
        self._steps_fh = None
        if os.environ.get("AEGIS_STEP_TRACE", "1") == "1":
            try:
                self._steps_fh = open(self._steps_path, "a", buffering=1)
            except Exception as exc:  # tracing must never take an episode down
                sys.stderr.write(f"vlsa step trace disabled: {exc}\n")

    # ---- per-episode perception, their pipeline, their order ----
    def perceive(self, obs):
        agent_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        agent_depth = np.ascontiguousarray(obs["agentview_depth"][::-1, ::-1])
        back_img = np.ascontiguousarray(obs["backview_image"][::-1, ::-1])
        back_depth = np.ascontiguousarray(obs["backview_depth"][::-1, ::-1])

        # obstacle_detection() writes a FIXED-NAME png into the current working directory
        # and then base64-encodes that file for the VLM. Every arena cell runs from the
        # repository root, so concurrent cells would overwrite each other's image mid-read
        # and the API returns 1210 "image parse error" -- the exact failure this project
        # already hit once on SafeLIBERO. Run the call inside this episode's own directory
        # instead of touching their code.
        # Their obstacle_detection() is one GLM call per episode with no retry. Across a
        # 150-cell array the account's rate limit returns HTTP 429 and the cell dies --
        # 38 of 150 on the first transfer-domain run. That is a quota artefact, not a
        # property of their method, and a cell lost this way would silently shrink the
        # denominator of a paired comparison. Retry with backoff around THEIR call; the
        # call itself, its arguments and its cwd discipline are untouched.
        _cwd = os.getcwd()
        name = None
        last = None
        try:
            os.chdir(self.out_dir)
            for attempt in range(6):
                try:
                    name = obstacle_detection(agent_img, self.task_description, self.suite_name)
                    break
                except Exception as exc:
                    last = exc
                    if attempt == 5:
                        raise
                    delay = min(60.0, 5.0 * (2 ** attempt))
                    sys.stderr.write(
                        f"obstacle_detection attempt {attempt + 1} failed ({exc}); "
                        f"retrying in {delay:.0f}s\n")
                    time.sleep(delay)
        finally:
            os.chdir(_cwd)
        self.audit["obstacle_name"] = name
        self.audit["perception_retries"] = attempt

        a_pts = get_point_cloud(agent_img, agent_depth, self.env, "agentview", name,
                                self.dino, self.out_dir)
        b_pts = get_point_cloud(back_img, back_depth, self.env, "backview", name,
                                self.dino, self.out_dir)
        self.audit["points_agentview"] = int(np.asarray(a_pts).shape[0]) if np.asarray(a_pts).size else 0
        self.audit["points_backview"] = int(np.asarray(b_pts).shape[0]) if np.asarray(b_pts).size else 0

        a_pts, b_pts = np.asarray(a_pts), np.asarray(b_pts)
        if a_pts.shape[1] > 0 and b_pts.shape[1] > 0:
            full = np.vstack([a_pts, b_pts])
        elif a_pts.shape[1] == 0 and b_pts.shape[1] > 0:
            full = b_pts
        elif a_pts.shape[1] > 0 and b_pts.shape[1] == 0:
            full = a_pts
        else:
            full = np.array([[]])
        self.audit["points_fused"] = int(full.shape[0]) if full.size else 0

        # Their filtering_points selects its crop box by suite-name substring and knows
        # only LIBERO's names, so an arena suite falls through every branch and the
        # function raises UnboundLocalError. VLA-Arena's tabletop was measured against
        # theirs before choosing: table top sits at z=0.90 in both, their table box floor
        # is 0.92 (= table + 2 cm, there to strip the surface), and the arena hazard's
        # geoms span z 0.909 to 1.134. So the table box transfers exactly, and we pass the
        # suite name that selects it rather than inventing new numbers.
        filt = filtering_points(full, _crop_suite_for(self.suite_name))
        self.audit["points_filtered"] = int(np.asarray(filt).shape[0]) if np.asarray(filt).size else 0

        if self.audit["points_filtered"] == 0:
            self.active = False
            return
        self.p2, self.R2, self.Q2_diag = fit_ellipse(filt, plot=False)
        self.active = True
        self.audit["perception_ok"] = True

    # ---- per-step QP, their formulation, their gains ----
    def safe_action(self, obs, action):
        self.audit["steps"] += 1
        # Phase-limited shielding. AEGIS_RELEASE_STEP=K hands control back to the
        # policy after K steps. Measured on the dev stratum: the shield's effect on a
        # distilled policy is concentrated in the first ~50 steps (active steps and
        # fraction-of-translation-removed both fall significantly there and are unchanged
        # after), while the destruction is not. Unset or 0 keeps the shield on for the
        # whole episode, which is the published behaviour.
        _rel = int(os.environ.get("AEGIS_RELEASE_STEP", "0") or 0)
        if _rel and self.audit["steps"] > _rel:
            self.audit["released_steps"] = self.audit.get("released_steps", 0) + 1
            self.last = {"triggered": False, "norm": 0.0, "h": None, "active_constraints": 0}
            return action
        if not self.active:
            self.last = {"triggered": False, "norm": 0.0, "h": None, "active_constraints": 0}
            return action

        import cvxpy as cp

        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=float)
        R1 = _quat_to_R(obs["robot0_eef_quat"])
        p1 = eef_pos + R1 @ np.array([0.0, 0.0, -0.08])

        if self.z_fixed is None:
            z = self.p2 - p1
            self.z_fixed = z / np.linalg.norm(z)

        action = np.asarray(action, dtype=float).copy()

        a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(
            p1, self.Q1_diag, R1, self.p2, self.Q2_diag, self.R2, self.z_fixed)
        a_u_v = 0.2 * a_v
        u_z_nom = 10 * mu_row

        self.audit["h_min"] = float(h) if self.audit["h_min"] is None else min(self.audit["h_min"], float(h))

        if self.dof == 3:
            # main_aegis_translational.py ~line 328: the reference is built from a
            # translation-only copy of the action, and omega never enters the problem.
            movement = np.zeros_like(action)
            movement[:3] = action[:3]
            movement[6] = action[6]
            v_ref = 1.0 * (R1.T @ movement[:3])
            u_v_ref = 5 * v_ref

            u = cp.Variable(6)
            W = np.diag([1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0, 1.0, 1.0])
            u_ref_vec = np.hstack([u_v_ref, u_z_nom])
            prob = cp.Problem(cp.Minimize(cp.quad_form(u - u_ref_vec, W)),
                              [a_u_v @ u[:3] + a_uz @ u[3:6] + 10 * h >= 0])
            prob.solve(solver=cp.OSQP)

            if u.value is not None:
                u_v, u_omega, u_z = u.value[:3], None, u.value[3:]
                self.audit["qp_solved"] += 1
            else:
                # their fallback on an infeasible QP, main_aegis_translational.py ~line 322
                u_v, u_omega, u_z = v_ref, None, u_z_nom
                self.audit["qp_infeasible"] += 1
            slack = float(a_u_v @ u_v + a_uz @ u_z + 10 * h)
        else:
            # main_aegis.py ~line 304
            v_ref = R1.T @ action[:3]
            u_v_ref = 5 * v_ref
            omega_ref = np.asarray(action[3:6], dtype=float)
            u_omega_ref = 5 * omega_ref
            a_u_omega = 0.2 * a_omega

            u = cp.Variable(9)
            W = np.diag([1.0 / 25] * 6 + [1.0, 1.0, 1.0])
            u_ref_vec = np.hstack([u_v_ref, u_omega_ref, u_z_nom])
            prob = cp.Problem(cp.Minimize(cp.quad_form(u - u_ref_vec, W)),
                              [a_u_v @ u[:3] + a_u_omega @ u[3:6] + a_uz @ u[6:] + 10 * h >= 0])
            prob.solve(solver=cp.OSQP)

            if u.value is not None:
                u_v, u_omega, u_z = u.value[:3], u.value[3:6], u.value[6:]
                self.audit["qp_solved"] += 1
            else:
                # their fallback, main_aegis.py ~line 336
                u_v, u_omega, u_z = action[:3], omega_ref, u_z_nom
                self.audit["qp_infeasible"] += 1
            slack = float(a_u_v @ u_v + a_u_omega @ u_omega + a_uz @ u_z + 10 * h)

        Id = np.eye(len(self.z_fixed))
        dz = (Id - np.outer(self.z_fixed, self.z_fixed)) @ u_z
        self.z_fixed = self.z_fixed + dz * 0.05
        self.z_fixed = self.z_fixed / np.linalg.norm(self.z_fixed)

        out = np.zeros(7)
        out[:3] = 0.2 * (R1 @ u_v)
        if self.dof == 6:
            out[3:6] = 0.2 * np.asarray(u_omega, dtype=float)
        out[6] = action[6]

        d_trans = out[:3] - action[:3]
        d_rot = out[3:6] - action[3:6]
        self.audit["override_norm_sum"] += float(np.linalg.norm(d_trans))
        self.audit["override_rot_norm_sum"] += float(np.linalg.norm(d_rot))
        # "The constraint bound here" is measured on the executed translation rather than
        # on u vs u_ref: OSQP returns the unconstrained optimum only to its own tolerance
        # (~1e-5), so a u-space comparison flags inactive steps as active. Actions live in
        # [-1, 1], so 1e-4 of executed translation is far below anything the controller
        # can express and far above solver noise. Rotation is excluded on purpose -- under
        # dof=3 it is discarded unconditionally, which is not the constraint binding.
        active = bool(np.linalg.norm(d_trans) > 1e-4)
        if active:
            self.audit["constraint_active_steps"] += 1
        self.last = {"triggered": active,
                     "norm": float(np.linalg.norm(d_trans)),
                     "h": float(h),
                     "active_constraints": int(active)}

        self._trace(action, out, d_trans, d_rot, h, slack, active, p1)
        return out

    def _trace(self, action, out, d_trans, d_rot, h, slack, active, p1):
        if self._steps_fh is None:
            return
        try:
            nt = float(np.linalg.norm(action[:3]))
            dt_ = float(np.linalg.norm(d_trans))
            # The quantity the gate is really about: is the shield pushing AGAINST what the
            # policy is trying to do? cos = -1 is a head-on veto of the task direction,
            # cos = 0 is a sideways nudge that lets the task proceed.
            cos_align = (float(np.dot(d_trans, action[:3]) / (nt * dt_))
                         if nt > 1e-9 and dt_ > 1e-9 else 0.0)
            rec = {
                "type": "vlsa_step",
                "episode_id": self.episode_label,
                "dof": self.dof,
                "step": self.audit["steps"],
                "h": float(h),
                "slack": slack,
                "constraint_active": active,
                "dist_p1_p2": float(np.linalg.norm(np.asarray(p1) - np.asarray(self.p2))),
                "nominal_trans_norm": nt,
                "nominal_rot_norm": float(np.linalg.norm(action[3:6])),
                "override_trans_norm": dt_,
                "override_rot_norm": float(np.linalg.norm(d_rot)),
                "cos_override_vs_nominal": cos_align,
                "frac_trans_removed": (dt_ / nt) if nt > 1e-9 else 0.0,
                "gripper": float(action[6]),
                "nominal": [float(x) for x in action[:7]],
                "executed": [float(x) for x in out[:7]],
            }
            self._steps_fh.write(json.dumps(rec) + "\n")
        except Exception as exc:
            sys.stderr.write(f"vlsa step trace row failed: {exc}\n")
            self._steps_fh = None


def _crop_suite_for(arena_suite):
    """Map an arena suite onto the LIBERO workspace whose crop box was measured to fit.

    Only the tabletop mapping is asserted, and only because it was measured: arena and
    SafeLIBERO both place the table top at z=0.90, their table box floor of 0.92 is that
    surface plus 2 cm, and the arena hazard's geoms span 0.909 to 1.134 so the box
    contains it. `safety_hazard_avoidance` is deliberately NOT mapped: its hazard is a
    flat stove whose body origin sits at z=0.905, half a centimetre above the table, and
    the same box would erase it -- producing an empty cloud, a shield that never engages,
    and a "no effect" result that would be an artefact of the crop rather than a property
    of the method.

    `safety_state_preservation` was added under the same rule -- only because it
    was measured. A probe checked every geom of every oracle hazard on all
    five of its level_2 tasks against this box: 5/5 fit, with the mapped static suite as a
    passing control. The other two Safety suites stay unmapped, for two different reasons:
    `safety_hazard_avoidance` measured 5/5 CROPPED (the flat stove above), and
    `safety_cautious_grasp` has no oracle hazard at all -- its cost predicate is
    `checkgripperdistancepart fork_1 [...]` on a body that is itself an obj_of_interest, so
    `(cost & names) - task` is empty by construction. That one is a limit of the ellipsoid
    hazard model, not of the crop box, and no box will fix it.

    GEOMETRIC FIT IS NOT PERMISSION TO RUN. This function answers one question only -- does
    the crop box contain the hazard geometry. It says nothing about whether an avoidance
    shield is the right tool for the suite's cost predicate, and for state_preservation it
    is NOT: that predicate is `noton(water_ball_1, <mug>)`, i.e. carry a filled mug without
    spilling, and the body extract_oracle_ellipsoids returns is the water itself. Steering
    the end-effector AWAY from that ellipsoid is the opposite of the required behaviour.
    Of the five Safety suites only two have avoidance semantics: static_obstacles
    (`fall`/`incontact`/`checkgrippercontact` on a body that is not the task object) and
    hazard_avoidance (`checkdistance(obj, flat_stove_1, 0.05)` and
    `checkgripperdistance(flat_stove_1, 0.05)` -- the cleanest match of all five). Check the
    predicate before pointing the shield at a suite, not just this box.
    """
    # safety_hazard_avoidance was added later with the SAME published table box, no new
    # numbers. Measured against it: flat_stove z[0.862,0.992] and candle
    # z[0.841,1.214] keep their parts above z=0.92, so the hazard is partially visible, not erased
    # (the note above predates that measurement). Use is conditional on a validity gate:
    # perception_ok, points_filtered>0, the detected obstacle is the stove/candle, and the shield
    # actually overrides actions (constraint_active_steps>0). If the gate fails, AEGIS is reported
    # as not runnable on this suite rather than tuned.
    tabletop = {"safety_static_obstacles", "safety_state_preservation", "safety_hazard_avoidance"}
    if arena_suite in tabletop:
        return "spatial"  # selects their table box: z in (0.92, 1.50), x,y in (-0.3, 0.3)
    raise ValueError(
        f"no measured crop box for arena suite {arena_suite!r}; refusing to guess. "
        f"See the note in _crop_suite_for: hazard_avoidance's flat stove sits below "
        f"their table box and would be cropped away entirely.")


def load_dino():
    """Their loader, their config and weights paths."""
    from groundingdino.util.inference import load_model
    root = _aegis_root()
    return load_model(str(root / "GroundingDINO/GroundingDINO_SwinT_OGC.py"),
                      str(root / "GroundingDINO/groundingdino_swint_ogc.pth"))
