#!/usr/bin/env bash
#$ -N s1_conf
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-50
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/step1_confirm/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/step1_confirm/batch/j.$JOB_ID.$TASK_ID.out
# Step1: confirm the 5 Aegis gaps found in Step0 are reproducible across init states.
# 5 gap configs x {base,shield} x 5 init offsets = 50 runs.
# Key questions: (a) does the shield-induced REGRESSION on static L2 t3/t4 replicate?
#                (b) is the shield cost real policy-induced, or initial-state attributed?
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers PI05_SERVER_TIMEOUT_SEC=600
cd ${SE_VLA_ROOT}

ARMS=(base shield)
# suite:level:task  -- only the 5 configs that showed a gap in Step0
CONFIGS=(
  safety_dynamic_obstacles:1:4
  safety_static_obstacles:2:0
  safety_static_obstacles:2:1
  safety_static_obstacles:2:3
  safety_static_obstacles:2:4
)
i=$((SGE_TASK_ID-1))
ARM=${ARMS[$((i/25))]}
j=$((i%25))
CFG=${CONFIGS[$((j/5))]}
OFF=$((j%5))
SUITE=${CFG%%:*}; rest=${CFG#*:}; LEVEL=${rest%%:*}; TASK=${rest##*:}
SEED=23

BASE_CKPT=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
OUT=${WORK_ROOT}/lora_dagger/step1_confirm/${ARM}/${SUITE}_L${LEVEL}_t${TASK}_off${OFF}
mkdir -p "$OUT" ${WORK_ROOT}/lora_dagger/step1_confirm/batch
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED="$SEED" SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_POLICY_CHECKPOINT_DIR="$BASE_CKPT"

if [ "$ARM" = "shield" ]; then
  export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1
  export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
else
  export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0
fi

echo "=== Step1 arm=$ARM suite=$SUITE L$LEVEL task=$TASK off=$OFF ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite "$SUITE" \
  --task-level "$LEVEL" --task-id "$TASK" --offset "$OFF" --seed "$SEED" \
  --replan-steps 1 --trials 1 --arm base --port-base $((34000 + SGE_TASK_ID)) \
  --array-task-id "$SGE_TASK_ID" --server-attempts 3
echo "STEP1_EXIT=$? arm=$ARM $SUITE L$LEVEL t$TASK off=$OFF"
