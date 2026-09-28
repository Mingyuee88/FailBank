#!/usr/bin/env bash
#$ -N v8_census
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-10
#$ -tc 5
#$ -o ${WORK_ROOT}/lora_dagger/v8_census/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v8_census/batch/j.$JOB_ID.$TASK_ID.out
#
# v8_census -- close the last 5 holes in the L1t2 deployment census.
#
# The winning arm (newkey-h5, no gate) has been deployed on 42 of the 47 usable
# offsets: all 37 base-success offsets plus the 5 base-failure probe offsets
# (29,36,38,42,43).  The 5 base-failure offsets that were used for ADAPTATION
# (0,5,10,26,27) were never deployment-evaluated.  Running them turns the
# whole-task number from an extrapolation into a complete 47/47 census.
#
# These 5 are the offsets the memory was harvested FROM, so a success here is
# training-set performance, NOT evidence of generalisation.  Reported separately
# from the held-out fix rate (1/5 on 29,36,38,42,43), never pooled with it.
#
# Also runs the same 5 offsets under --arm base through THIS harness, so the
# base reference for them is not borrowed from the pi05_static_sweep scan.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

OFFSETS=(0 5 10 26 27)
i=$((SGE_TASK_ID - 1))
if [ $i -lt 5 ]; then MODE=memory; OFF=${OFFSETS[$i]}; else MODE=base; OFF=${OFFSETS[$((i - 5))]}; fi

ARM_DIR=newkey-h5
OUT=$LR/v8_census/$MODE/off$OFF
mkdir -p "$OUT" $LR/v8_census/batch

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="census_${MODE}_off${OFF}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
# byte-for-byte the key used by the winning arm
export SE_VLA_RELATION_MEMORY_FEATURE_KEYS="cost_pair_min_distance,cost_pair_delta_xy[0],cost_pair_delta_xy[1],eef_hazard_distance,eef_target_distance,episode_step_index"

PORT=$((48000 + 30 * SGE_TASK_ID))
if [ "$MODE" = memory ]; then
  CK=$(ls -d $LR/v7_l1t2/$ARM_DIR/round_*/settled_checkpoint.json 2>/dev/null | tail -1)
  [ -n "$CK" ] || { echo "no settled checkpoint for $ARM_DIR"; exit 1; }
  echo "=== census memory off=$OFF ckpt=$CK ==="
  PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
    roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
    --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
    --replan-steps 1 --trials 1 --arm pruned_memory --checkpoint "$CK" \
    --port-base $PORT --array-task-id 1 --server-attempts 3
else
  echo "=== census base off=$OFF ==="
  PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
    roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
    --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
    --replan-steps 1 --trials 1 --arm base \
    --port-base $PORT --array-task-id 1 --server-attempts 3
fi
echo "CENSUS_EXIT=$? mode=$MODE off=$OFF"
