#!/usr/bin/env bash
#$ -N p3b_ret
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-370
#$ -tc 14
#$ -o ${WORK_ROOT}/lora_dagger/phase3b_retention/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/phase3b_retention/batch/j.$JOB_ID.$TASK_ID.out
#
# Phase3b -- the untested half of the 7/22 hard gate: does shield-off LoRA
# distillation BREAK the offsets where base already succeeds?
# Phase3a only ever used the 9 base-FAILURE offsets, so "does not break existing
# successes" has never been measured.
#
# Cells: 37 base-always-success offsets (from the completed phase0_base_scan,
# 141/141 results) x {matched base baseline, each of the 9 LOSO fold adapters}.
# n=1 per cell is justified: across the 141-run scan, SR was identical for all
# 3 nominal seeds at every one of the 47 offsets (--seed is a no-op in this
# pipeline; it only feeds select_init_state_index with offset_random=False).
#
# All arms run --arm base with the shield OFF; the LoRA effect comes from
# SE_VLA_POLICY_CHECKPOINT_DIR pointing at that fold's pre-merged checkpoint.
# No folding here -- phase2_training/merged/offset_<F> already exists (105 GB),
# so this job writes no large intermediates.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers PI05_SERVER_TIMEOUT_SEC=600
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
cd ${SE_VLA_ROOT}

ROOT=${WORK_ROOT}/lora_dagger/phase3b_retention
LINE=$(sed -n "${SGE_TASK_ID}p" "$ROOT/jobs.txt")
[ -n "$LINE" ] || { echo "no job for task $SGE_TASK_ID"; exit 0; }
ARM=$(echo "$LINE" | awk '{print $1}')
OFF=$(echo "$LINE" | awk '{print $2}')
FOLD=$(echo "$LINE" | awk '{print $3}')

BASE_CKPT=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
if [ "$ARM" = "lora" ]; then
  CKPT=${WORK_ROOT}/lora_dagger/phase2_training/merged/offset_${FOLD}
  OUT=$ROOT/lora/off${OFF}_f${FOLD}
else
  CKPT=$BASE_CKPT
  OUT=$ROOT/base/off${OFF}
fi
[ -d "$CKPT" ] || { echo "MISSING CKPT $CKPT"; exit 1; }
mkdir -p "$OUT"

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
# hard-required by run.py even with aegis off
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_SRD_WRITES_ENABLED=0
unset SE_VLA_PHASE1_RECORD || true
export SE_VLA_POLICY_CHECKPOINT_DIR="$CKPT"
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0

echo "=== Phase3b arm=$ARM offset=$OFF fold=$FOLD ckpt=$CKPT ==="
date +%s
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm base --port-base 40000 \
  --array-task-id "$SGE_TASK_ID" --server-attempts 3
echo "P3B_EXIT=$? arm=$ARM off=$OFF fold=$FOLD"
date +%s
