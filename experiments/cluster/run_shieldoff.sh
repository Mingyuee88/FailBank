#!/usr/bin/env bash
#$ -N shoff
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-20
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/shieldoff/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/shieldoff/batch/j.$JOB_ID.$TASK_ID.out
# Shield-OFF baseline on the collection offsets 10-29, seed 17.
#
# Round 0 found that on 3 of 5 offsets the task SUCCEEDS with the shield off and
# FAILS with it on (cost 209-317), i.e. the collisions we have been studying are
# largely shield-induced. That was n=5. This pairs a shield-off run with every
# shield-on episode already collected at seed 17 on offsets 10-29, so the claim
# can be checked at n=20 on matched initial states.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers PI05_SERVER_TIMEOUT_SEC=600
cd ${SE_VLA_ROOT}

OFF=$((9 + SGE_TASK_ID))     # 10..29
SEED=17
ROOT=${WORK_ROOT}/lora_dagger/shieldoff
OUT=$ROOT/off${OFF}_s${SEED}
mkdir -p "$OUT" "$ROOT/batch"

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED="$SEED" SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="shoff_off${OFF}_s${SEED}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
# shield fully off
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0

echo "=== shield-off off=$OFF seed=$SEED ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 2 --task-id 4 --offset "$OFF" --seed "$SEED" \
  --replan-steps 1 --trials 1 --arm base --port-base $((41000 + SGE_TASK_ID)) \
  --array-task-id "$SGE_TASK_ID" --server-attempts 3
echo "SHOFF_EXIT=$? off=$OFF"
