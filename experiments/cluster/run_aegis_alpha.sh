#!/usr/bin/env bash
#$ -N aeg_a
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-20
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/aegis_alpha/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/aegis_alpha/batch/j.$JOB_ID.$TASK_ID.out
# Re-run the Aegis baseline at the PAPER's parameters before comparing anything.
#
# arXiv:2512.11891 (VLSA/AEGIS) specifies a linear class-K function alpha(h)=10h.
# Every Aegis number we produced so far used alpha=3 with eef/obstacle radii
# 0.03/0.04 and margin 0.02, i.e. a substantially MORE conservative layer than the
# paper's (the constraint normal.v >= -alpha*h binds less often as alpha grows).
# Those runs therefore characterise a mis-parameterised Aegis, not the method.
#
# A1: alpha=10 (paper), code-default geometry 0.06/0.08/0.0
# A2: alpha=1  (code default), same geometry -- brackets the paper value
# Tasks: L2_t4 (main battlefield) and L2_t3 (the claimed shield regression).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers PI05_SERVER_TIMEOUT_SEC=600
cd ${SE_VLA_ROOT}

TASKS=(3 4)
ALPHAS=(10 1)
i=$((SGE_TASK_ID-1))
TASK=${TASKS[$((i/10))]}
rem=$((i%10))
ALPHA=${ALPHAS[$((rem/5))]}
OFF=$((rem%5))
SUITE=safety_static_obstacles; LEVEL=2; SEED=23

ROOT=${WORK_ROOT}/lora_dagger/aegis_alpha
BASE_CKPT=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
OUT=$ROOT/a${ALPHA}/${SUITE}_L${LEVEL}_t${TASK}_off${OFF}
mkdir -p "$OUT" "$ROOT/batch"

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED="$SEED" SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_POLICY_CHECKPOINT_DIR="$BASE_CKPT"

export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1
export SE_VLA_AEGIS_ALPHA="$ALPHA"
export SE_VLA_AEGIS_EEF_RADIUS=0.06 SE_VLA_AEGIS_ORACLE_RADIUS=0.08 SE_VLA_AEGIS_MARGIN=0.0

echo "=== aegis alpha=$ALPHA L${LEVEL} t${TASK} off=$OFF ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite "$SUITE" \
  --task-level "$LEVEL" --task-id "$TASK" --offset "$OFF" --seed "$SEED" \
  --replan-steps 1 --trials 1 --arm base --port-base $((37000 + SGE_TASK_ID)) \
  --array-task-id "$SGE_TASK_ID" --server-attempts 3
echo "AEGIS_ALPHA_EXIT=$? alpha=$ALPHA t$TASK off=$OFF"
