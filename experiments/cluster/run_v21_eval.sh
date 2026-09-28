#!/usr/bin/env bash
#$ -N v21_eval
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 2
#$ -t 1-188
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v21_eval/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v21_eval/batch/j.$JOB_ID.$TASK_ID.out
#
# v21_eval -- deploy the four self-evolved policies. No shield, no memory, no
# runtime wrapper: the weights are the method.
#
# This is what the external-memory line could never be. A read-only kNN store
# leaves the policy untouched, so removing it returns the robot to its original
# mistakes -- and today's permutation control showed its content was doing no
# measurable work anyway. Here the correction lives in the parameters, so the
# question "did the base policy improve" is finally the question being asked.
#
# ARMS (all four required; see v19_train for the preregistration):
#   A  old records, quiet_weight 0.0   should reproduce ~88% retention, ~40% fix
#   B  old records, quiet_weight 0.3   should reproduce the quiet-sweep null
#   C  new records, quiet_weight 0.3   the change under test
#   D  new records, quiet_weight 0.0   separates negative DATA from quiet REWEIGHT
#
# Verified before training, not assumed: A/B folds carry 0 quiet records drawn from
# success episodes, C/D carry 1228. The teacher set genuinely never contained a
# single state from a successful grasp until now.
#
# READOUT. Fix rate and retention rate are reported separately and never netted --
# a net dSR of 0 can be "nothing happened" or "one repair paid for by one new
# failure". obj_travel is reported alongside as the degenerate-solution guard, in
# BOTH directions: an arm that stops moving and an arm that flings the object both
# score a flattering CC.
#
# ★ Retention here is measured on all 47 offsets, but 18 of the base-success ones
# were used as negative training data (v18_split/split.json fixes which). The
# analysis MUST split retention into seen-in-training and held-out; only the
# held-out half is evidence. The final claim rests on the per-task transfer to
# L1t1/L1t3/L1t4, not on anything measured here.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

ARMS=(A_old_qw0.0 B_old_qw0.3 C_new_qw0.3 D_new_qw0.0)
OFFSETS=(0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47)
NOFF=47
i=$((SGE_TASK_ID - 1))
ARM=${ARMS[$((i / NOFF))]}
OFF=${OFFSETS[$((i % NOFF))]}

CK=$LR/v20_fold/$ARM/offset_0
[ -d "$CK" ] || { echo "missing folded checkpoint $CK"; exit 1; }

OUT=$LR/v21_eval/$ARM/off$OFF
mkdir -p "$OUT" $LR/v21_eval/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="v21_${ARM}_off${OFF}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
# the whole point: the evolved WEIGHTS are loaded, and nothing else is on
export SE_VLA_POLICY_CHECKPOINT_DIR="$CK"
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0

echo "=== v21 arm=$ARM off=$OFF ckpt=$CK on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((6000 + 15 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
echo "EVAL_EXIT=$? arm=$ARM off=$OFF"
