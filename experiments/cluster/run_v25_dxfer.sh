#!/usr/bin/env bash
#$ -N v25_dxfer
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_B
#$ -pe smp 2
#$ -t 1-150
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v25_dxfer/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v25_dxfer/batch/j.$JOB_ID.$TASK_ID.out
#
# v22_axfer -- does the self-evolved policy survive contact with unseen objects?
#
# WHY THIS ARM. The four-arm ablation closed and arm D -- the negative teacher data
# WITHOUT the quiet reweighting -- is the best of the four by a wide margin:
#
#     arm                       fix        held-out retention   dCC       net SR
#     A old data, qw 0.0        5/11 45.5%   16/18  88.9%       -26.4%    38/47
#     B old data, qw 0.3        4/11 36.4%   16/18  88.9%       -19.7%    37/47
#     C new data, qw 0.3        5/11 45.5%   13/18  72.2%        -6.3%    37/47
#     D new data, qw 0.0        7/11 63.6%   18/18 100.0%       -47.2%    41/47
#
# The 2x2 separates cleanly: the negative DATA helps on all three measures, the
# quiet REWEIGHTING hurts, and it hurts more when there is more quiet data for it
# to mis-weight. The original diagnosis (the teacher set contained literally zero
# states from a successful grasp) was right; feeding those states through
# quiet_weight was the wrong way to use them. Arm C alone would have read as a
# refutation -- which is exactly why all four arms were preregistered.
#
# WHAT IS STILL UNTESTED. All of that is one arena, n=11 failures and n=18 held-out
# successes, McNemar p=0.180. The external-memory line looked strongest on this
# same development domain and then collapsed on transfer. Arm A already transferred
# without collapse (fix 23/34, retention 98/116, net +5) -- this run asks whether
# D's larger within-task margin survives the same test.
#
# READOUT, fixed before submission: per-task fix rate and retention rate with
# Wilson intervals, never netted and never pooled into a single binomial (episodes
# cluster by task -- pooling hid a task being significantly harmed in v13_xfer);
# paired dCC; obj_travel ratio flagged in both directions; exact McNemar per task.
#
# The comparison that matters is against the SHIELD's transfer profile, not against
# zero. That profile is not yet measured on these tasks and is the natural follow-up
# if this arm holds.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

ARM=D_new_qw0.0
CK=$LR/v20_fold/$ARM/offset_0
[ -d "$CK" ] || { echo "missing folded checkpoint $CK"; exit 1; }

TASKS=(1 3 4)
NOFF=50
i=$((SGE_TASK_ID - 1))
TID=${TASKS[$((i / NOFF))]}
OFF=$((i % NOFF))

OUT=$LR/v25_dxfer/L1t${TID}/off${OFF}
mkdir -p "$OUT" $LR/v25_dxfer/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="dxfer_L1t${TID}_off${OFF}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
# the evolved weights are the entire method: no shield, no memory, no wrapper
export SE_VLA_POLICY_CHECKPOINT_DIR="$CK"
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0

echo "=== v25 arm=$ARM L1t${TID} off=$OFF ckpt=$CK on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id "$TID" --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((46000 + 20 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
echo "DXFER_EXIT=$? L1t$TID off=$OFF"
