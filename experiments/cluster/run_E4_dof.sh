#!/usr/bin/env bash
#$ -N E4_dof
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 4
#$ -t 1-94
#$ -tc 2
#$ -o ${WORK_ROOT}/E4_dof/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/E4_dof/batch/j.$JOB_ID.$TASK_ID.out
#
# E4 -- separate "the shield destroys episodes with no cost to remove" from
#       "the shield destroys episodes whose task needs wrist rotation".
#
# E3 ran main_aegis_translational.py, their `Ours_t` file. It builds
#     action_input = zeros(7); action_input[:3] = 0.2*R1@u_v; action_input[6] = gripper
# so the rotation channel is discarded at EVERY step, active constraint or not. On the 47
# arena offsets that arm destroyed 9 of 36 base successes and repaired 8 of 11 base
# failures. Those two sets are precisely {base succeeded} and {base failed}, so the
# barrier/cost-misalignment account and the amputated-rotation account predict the SAME
# partition and no observational analysis on the existing runs can tell them apart --
# measured: rotation demand separates destroyed from repaired at AUC 0.81, and separates
# base-success from base-failure at AUC 0.74 in the same direction, because they are the
# same contrast.
#
# The intervention that separates them is to restore the channel. main_aegis.py -- their
# own full formulation, same repository, same gains -- carries omega in the objective and
# in the constraint and emits action_input[3:6] = 0.2*u_omega. Under an inactive
# constraint it is the identity on both channels; the translational variant is the
# identity on translation and identically zero on rotation. Verified offline before
# submission (t_equiv.py): dof=3 reproduces the shipped port BIT-FOR-BIT over 200 steps
# (max|old-new| = 0.0, override_norm_sum and h_min equal to all digits), and dof=6 returns
# the nominal action to 5.6e-17 when the constraint is slack.
#
# If dof=6 still destroys those offsets, the mechanism claim is a property of the barrier
# and is now much stronger. If it does not, E3's headline was an artefact of the variant.
#
# Task 1-47  : dof=6, the arm that decides this.
# Task 48-94 : dof=3, a re-run of E3 under the identical code path. Two purposes -- it is
#              the determinism control (it must reproduce SR 35/47 and polCC 199 exactly,
#              same pinned host, and --seed is a no-op in this harness so nothing else can
#              account for a difference), and it collects the per-step trace E3 never
#              wrote, which is what any intervention gate has to be built and tested on.
#
# Everything except the shield variant is the machinery every other arm in this study ran
# through: same policy server, same offsets, same cost accounting, same result.json.
# Derived from run_E3_vlsa.sh by changing only the output root, the dof switch, the array
# length and the port base -- queue, host pin, gpu_card, smp, tc and all SRD_* values are
# unchanged from that verified recipe.
#
# The port base had to change and could not be inherited. server.py allocates
#     port = base_port + array_task_id + 1000 * attempt        (attempt = 0,1,2)
# and rejects anything above 65535, so a base above 63534 cannot survive its own retries.
# E3's `64000 + 20*SGE_TASK_ID` peaked at 64940, which is already past that bound on the
# third attempt and was only ever safe because no E3 cell needed to retry; carried into a
# 94-long array it exceeds 65535 on the FIRST attempt from task 77 up, and did -- 18 dof=3
# cells died with "allocated port out of range". Loudly, and with their result.json
# removed, so nothing silent got into the table. `50000 + 20*(SGE_TASK_ID-1)` peaks at
# 51860, stays disjoint per task across the whole array, and leaves the full retry budget.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
OUTROOT=${WORK_ROOT}/E4_dof

OFFS=(0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47)
N=${#OFFS[@]}
i=$((SGE_TASK_ID - 1))
if [ "$i" -lt "$N" ]; then
  DOF=6; ARM=dof6; j=$i
else
  DOF=3; ARM=dof3; j=$((i - N))
fi
[ "$j" -lt "$N" ] || { echo "no cell for task $SGE_TASK_ID"; exit 0; }
OFF=${OFFS[$j]}

OUT=$OUTROOT/$ARM/off$OFF
mkdir -p "$OUT" "$OUTROOT/batch"
if [ -s "$OUT/result.json" ] && \
   external/VLA-Arena/envs/openpi/.venv/bin/python -c \
     'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="pass" else 1)' \
     "$OUT/result.json" 2>/dev/null; then echo "SKIP $OUT"; exit 0; fi
rm -f "$OUT/result.json" "$OUT/vlsa_steps.jsonl"
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

# The VLM key is read at run time from a 600-mode file; it never enters a script or argv.
[ -r "$HOME/.config/zhipu_api_key" ] && export ZHIPUAI_API_KEY="$(cat "$HOME/.config/zhipu_api_key")"

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
unset SE_VLA_POLICY_CHECKPOINT_DIR || true
export SE_VLA_SHIELD_IMPL=vlsa_aegis
export SE_VLA_VLSA_DOF=$DOF
export SE_VLA_VLSA_STEP_TRACE=1 SE_VLA_VLSA_STEPS_PATH="$OUT/vlsa_steps.jsonl"
export SE_VLA_ARENA_SUITE=safety_static_obstacles
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="e4_${ARM}_off${OFF}"

echo "=== E4 published-AEGIS dof=$DOF off=$OFF on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((50000 + 20 * (SGE_TASK_ID - 1))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "E4_EXIT=$rc off=$OFF dof=$DOF"

if [ ! -s "$OUT/result.json" ] || ! external/VLA-Arena/envs/openpi/.venv/bin/python -c \
     'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="pass" else 1)' \
     "$OUT/result.json" 2>/dev/null; then
  echo "E4_CELL_FAILED off=$OFF dof=$DOF -- removing for retry"; rm -f "$OUT/result.json"; exit 1
fi

# The port must be shown to have PERCEIVED and SOLVED, not merely to have run. A silent
# degradation -- no obstacle named, empty cloud, every QP infeasible -- would otherwise
# look like a clean null result for their method. The dof and the step trace are asserted
# too: a dof=6 cell that silently ran the translational path would answer the wrong
# question while looking perfectly healthy.
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$OFF" "$DOF" <<'PYCHK'
import json, os, sys
out, off, dof = sys.argv[1], sys.argv[2], int(sys.argv[3])
audit = None
for line in open(os.path.join(out, "srd.jsonl")):
    try: r = json.loads(line)
    except Exception: continue
    if r.get("type") == "vlsa_audit":
        audit = r
assert audit is not None, "no vlsa_audit row -- the published shield never ran"
assert audit.get("dof") == dof, f"cell ran dof={audit.get('dof')}, expected {dof}"
r = json.load(open(os.path.join(out, "result.json")))
md = r.get("metric_decomposition", {})
steps_path = os.path.join(out, "vlsa_steps.jsonl")
n_steps = sum(1 for _ in open(steps_path)) if os.path.exists(steps_path) else 0
print(f"E4_RESULT off={off} dof={dof} sr={r.get('successes')} polcc={md.get('policy_induced_cc')} "
      f"obstacle={audit['obstacle_name']!r} steps={audit['steps']} qp_ok={audit['qp_solved']} "
      f"qp_infeasible={audit['qp_infeasible']} h_min={audit['h_min']} "
      f"override_sum={audit['override_norm_sum']:.4f} "
      f"override_rot_sum={audit.get('override_rot_norm_sum', float('nan')):.4f} "
      f"active_steps={audit.get('constraint_active_steps')} trace_rows={n_steps}")
assert audit["obstacle_name"], "the VLM named no obstacle"
assert audit["perception_ok"], "perception produced no ellipsoid -- shield never engaged"
assert audit["qp_solved"] > 0, "no QP ever solved"
assert audit["override_norm_sum"] > 0, "the shield never changed a single action"
assert n_steps >= audit["qp_solved"], f"step trace short: {n_steps} rows for {audit['qp_solved']} solves"
# dof=3 discards rotation unconditionally, so its rotation-override sum must be large;
# dof=6 carries rotation through, so under a slack constraint it must stay near zero.
if dof == 3:
    assert audit["override_rot_norm_sum"] > 0, "dof=3 did not drop rotation -- wrong code path"
print("E4_CELL_VERIFIED")
PYCHK
