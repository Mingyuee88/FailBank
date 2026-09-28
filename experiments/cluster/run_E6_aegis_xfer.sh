#!/usr/bin/env bash
#$ -N E6_axfer
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_B
#$ -pe smp 4
#$ -t 1-150
#$ -tc 1
#$ -o ${WORK_ROOT}/E6_axfer/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/E6_axfer/batch/j.$JOB_ID.$TASK_ID.out
#
# E6 -- the published AEGIS shield on the TRANSFER domain (L1t1 / L1t3 / L1t4, 50 offsets
# each), which is the one place the distilled arms have already been measured and the
# published shield has not.
#
# Why it is needed. Every "our method vs published AEGIS" number in this project lives on
# a single 47-offset stratum, and 28 of those 47 cells are in-sample for the distilled
# arms (18 base-successes used as negative training data, 9 base-failures that were the
# teacher's own collection offsets). On the 19 strictly held-out cells D beats the shield
# 19 to 12, paired p=0.0156 -- but two of those 19 are failures, so the repair half of the
# claim rests on two cells. A second domain is not a nicety here; it is what turns one
# thin significant result into a claim.
#
# The transfer arms already exist and are all pinned to HOST_B: base v11_merged,
# A v22_axfer, D v25_dxfer, our own controller v23_shxfer, 150 cells each. This adds the
# published shield on the identical cells and the identical host, so the comparison is
# paired the same way the dev-stratum one is. GPU model alone moves policy-induced cost
# 17.5% and flips about one cell in twelve, so running it anywhere else would not compare.
#
# dof=3 -- main_aegis_translational.py, their `Ours_t` file. E4 measured both variants on
# the dev stratum and the translational one is the stronger of the two there (35/47 @ 199
# against 29/47 @ 661), so this runs the baseline at its best rather than at its weakest.
#
# Config provenance: shield block from run_E4_dof.sh (derived from the verified
# run_E3_vlsa.sh); task/offset layout, host pin, seed and port formula from
# run_v25_dxfer.sh, so this arm is addressed to exactly the cells the other transfer arms
# ran. Nothing is invented.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
OUTROOT=${WORK_ROOT}/E6_axfer

TASKS=(1 3 4)
NOFF=50
i=$((SGE_TASK_ID - 1))
TID=${TASKS[$((i / NOFF))]}
OFF=$((i % NOFF))

OUT=$OUTROOT/L1t${TID}/off${OFF}
mkdir -p "$OUT" "$OUTROOT/batch"
if [ -s "$OUT/result.json" ] && \
   external/VLA-Arena/envs/openpi/.venv/bin/python -c \
     'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="pass" else 1)' \
     "$OUT/result.json" 2>/dev/null; then echo "SKIP $OUT"; exit 0; fi
rm -f "$OUT/result.json" "$OUT/vlsa_steps.jsonl"
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

[ -r "$HOME/.config/zhipu_api_key" ] && export ZHIPUAI_API_KEY="$(cat "$HOME/.config/zhipu_api_key")"

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
unset SE_VLA_POLICY_CHECKPOINT_DIR || true
export SE_VLA_SHIELD_IMPL=vlsa_aegis
export SE_VLA_VLSA_DOF=3
export SE_VLA_VLSA_STEP_TRACE=1 SE_VLA_VLSA_STEPS_PATH="$OUT/vlsa_steps.jsonl"
export SE_VLA_ARENA_SUITE=safety_static_obstacles
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="e6_axfer_L1t${TID}_off${OFF}"

echo "=== E6 published-AEGIS L1t${TID} off=$OFF on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id "$TID" --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((46000 + 20 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "E6_EXIT=$rc L1t$TID off=$OFF"

if [ ! -s "$OUT/result.json" ] || ! external/VLA-Arena/envs/openpi/.venv/bin/python -c \
     'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="pass" else 1)' \
     "$OUT/result.json" 2>/dev/null; then
  echo "E6_CELL_FAILED L1t$TID off=$OFF -- removing for retry"; rm -f "$OUT/result.json"; exit 1
fi

# These tasks use objects the shield's VLM prompt was never written for -- its candidate
# list is LIBERO's. A cell where perception quietly returned nothing would produce a shield
# that never engages and a flattering "no harm done" row, which is the single most
# misleading failure this arm could have. So perception and engagement are asserted, not
# hoped for, exactly as in E3/E4.
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$TID" "$OFF" <<'PYCHK'
import json, os, sys
out, tid, off = sys.argv[1], sys.argv[2], sys.argv[3]
audit = None
for line in open(os.path.join(out, "srd.jsonl")):
    try: r = json.loads(line)
    except Exception: continue
    if r.get("type") == "vlsa_audit":
        audit = r
assert audit is not None, "no vlsa_audit row -- the published shield never ran"
assert audit.get("dof") == 3, f"cell ran dof={audit.get('dof')}, expected 3"
r = json.load(open(os.path.join(out, "result.json")))
md = r.get("metric_decomposition", {})
sp = os.path.join(out, "vlsa_steps.jsonl")
n_steps = sum(1 for _ in open(sp)) if os.path.exists(sp) else 0
print(f"E6_RESULT L1t{tid} off={off} sr={r.get('successes')} polcc={md.get('policy_induced_cc')} "
      f"obstacle={audit['obstacle_name']!r} pts_filt={audit['points_filtered']} "
      f"steps={audit['steps']} qp_ok={audit['qp_solved']} qp_infeasible={audit['qp_infeasible']} "
      f"h_min={audit['h_min']} override_sum={audit['override_norm_sum']:.4f} "
      f"active_steps={audit.get('constraint_active_steps')} trace_rows={n_steps}")
assert audit["obstacle_name"], "the VLM named no obstacle"
assert audit["perception_ok"], "perception produced no ellipsoid -- shield never engaged"
assert audit["qp_solved"] > 0, "no QP ever solved"
assert audit["override_norm_sum"] > 0, "the shield never changed a single action"
assert n_steps >= audit["qp_solved"], f"step trace short: {n_steps} rows for {audit['qp_solved']} solves"
print("E6_CELL_VERIFIED")
PYCHK
