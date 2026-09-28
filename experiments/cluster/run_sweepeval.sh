#!/usr/bin/env bash
#$ -N swpeval
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-13
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/phase2_sweep_eval/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/phase2_sweep_eval/batch/j.$JOB_ID.$TASK_ID.out
# Closed-loop, shield-off evaluation of every saved checkpoint along three training
# trajectories. Answers the WITHIN-fold question that the cross-fold correlation
# (r=+0.538, p~0.14, confounded by offset difficulty) cannot: along ONE trajectory,
# does a lower held-out triggered loss go with a worse deployment outcome?
# Job list is a plain text file (fold step ckpt per line) read with sed, because
# embedding python quoting inside an ssh heredoc silently corrupted it once.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers PI05_SERVER_TIMEOUT_SEC=600
cd ${SE_VLA_ROOT}

LIST=${WORK_ROOT}/lora_dagger/phase2_sweep_eval/jobs.txt
LINE=$(sed -n "${SGE_TASK_ID}p" "$LIST")
FOLD=$(echo "$LINE" | awk '{print $1}')
STEP=$(echo "$LINE" | awk '{print $2}')
CKPT=$(echo "$LINE" | awk '{print $3}')
if [ -z "${CKPT:-}" ]; then echo "FATAL: empty job line $SGE_TASK_ID"; exit 1; fi

OUT=${WORK_ROOT}/lora_dagger/phase2_sweep_eval/fold${FOLD}_step${STEP}
mkdir -p "$OUT"
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_SRD_WRITES_ENABLED=0
unset SE_VLA_PHASE1_RECORD || true
export SE_VLA_POLICY_CHECKPOINT_DIR="$CKPT"
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0

echo "=== sweep eval fold=$FOLD step=$STEP ckpt=$CKPT ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$FOLD" --seed 23 \
  --replan-steps 1 --trials 1 --arm base --port-base $((45000 + SGE_TASK_ID)) \
  --array-task-id "$SGE_TASK_ID" --server-attempts 3
echo "SWEEPEVAL_EXIT=$? fold=$FOLD step=$STEP"
