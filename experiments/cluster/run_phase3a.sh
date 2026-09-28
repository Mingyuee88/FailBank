#!/usr/bin/env bash
#$ -N p3a_eval
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-81
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/phase3a/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/phase3a/batch/j.$JOB_ID.$TASK_ID.out
# Phase3a three-arm shield-off eval (offset generalization). 3 arms x 9 held-out
# offsets x 3 new seeds = 81. All arms run --arm base; the shield is env-driven and
# the LoRA effect comes from the folded checkpoint (SE_VLA_POLICY_CHECKPOINT_DIR).
#   base   : finetuned base, aegis OFF
#   shield : finetuned base, aegis ON (late_margin: a3 eef.03 or.04 m.02) = privileged upper bound
#   lora   : merged/offset_<X> (base+that fold's LoRA), aegis OFF = our method
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers PI05_SERVER_TIMEOUT_SEC=600
cd ${SE_VLA_ROOT}

ARMS=(base shield lora)
OFFSETS=(0 5 10 26 27 29 38 42 43)
SEEDS=(23 29 31)
i=$((SGE_TASK_ID-1))
ARM=${ARMS[$((i/27))]}
OFF=${OFFSETS[$(((i/3)%9))]}
SEED=${SEEDS[$((i%3))]}

BASE_CKPT=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
MERGED=${WORK_ROOT}/lora_dagger/phase2_training/merged/offset_${OFF}
OUT=${WORK_ROOT}/lora_dagger/phase3a/${ARM}/off${OFF}_seed${SEED}
mkdir -p "$OUT" ${WORK_ROOT}/lora_dagger/phase3a/batch
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED="$SEED" SE_VLA_RESULT_PATH="$OUT/result.json"

# SRD distance constants are hard-required by run.py even when aegis is OFF.
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_SRD_WRITES_ENABLED=0
# Phase3a: NO recorder.
unset SE_VLA_PHASE1_RECORD || true

if [ "$ARM" = "shield" ]; then
  export SE_VLA_POLICY_CHECKPOINT_DIR="$BASE_CKPT"
  export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1
  export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
elif [ "$ARM" = "lora" ]; then
  export SE_VLA_POLICY_CHECKPOINT_DIR="$MERGED"
  export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0
else  # base
  export SE_VLA_POLICY_CHECKPOINT_DIR="$BASE_CKPT"
  export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0
fi

echo "=== Phase3a arm=$ARM offset=$OFF seed=$SEED ckpt=$SE_VLA_POLICY_CHECKPOINT_DIR ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed "$SEED" \
  --replan-steps 1 --trials 1 --arm base --port-base $((32000 + SGE_TASK_ID)) \
  --array-task-id "$SGE_TASK_ID" --server-attempts 3
echo "ARM_EXIT=$? arm=$ARM off=$OFF seed=$SEED"
