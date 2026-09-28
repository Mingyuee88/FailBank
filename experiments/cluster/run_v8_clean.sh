#!/usr/bin/env bash
#$ -N v8_clean
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 2
#$ -t 1-94
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v8_clean/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v8_clean/batch/j.$JOB_ID.$TASK_ID.out
#
# v8_clean -- the first measurement of this method that is free of the hardware
# confound.  ONE node, all 47 usable offsets, both arms.
#
# v8_repeat established that with the node pinned the pipeline is bit-identical
# across reruns (11/11 offsets, SR and CC, 0.0% spread).  So a paired per-offset
# comparison run entirely on HOST_A has NO noise term at all: any difference
# between the two arms is the method, exactly.
#
# Every prior number for this method was collected under `-l h=(A6K_HOSTS*|L40S_HOSTS*)`
# with SGE scattering tasks across A6000 and L40S, and v8_host showed that swap
# alone moves polCC on 11/12 offsets (mean +17.5%) and flips SR on 1/12.  So the
# headline "dSR = 0.000, dCC = -3.7%" is not yet a real measurement -- this job
# makes it one.
#
# Arms:  base            frozen pi05, no memory, shield off
#        pruned_memory   same policy + the settled newkey-h5 checkpoint,
#                        shield off, read-only, gain 1, no abstain
#                        (the configuration the 08-10 sweep selected)
#
# 47 offsets x 2 arms = 94 cells.  Offsets 0-47 excluding 9 (which has no init
# state), matching phase0_base_scan's usable set.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

OFFSETS=(0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47)
NOFF=47
i=$((SGE_TASK_ID - 1))
OFF=${OFFSETS[$((i % NOFF))]}
if [ $((i / NOFF)) -eq 0 ]; then MODE=base; else MODE=memory; fi

OUT=$LR/v8_clean/$MODE/off$OFF
mkdir -p "$OUT" $LR/v8_clean/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="clean_${MODE}_off${OFF}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0

PORT=$((16000 + 25 * SGE_TASK_ID))
if [ "$MODE" = memory ]; then
  export SE_VLA_RELATION_MEMORY_FEATURE_KEYS="cost_pair_min_distance,cost_pair_delta_xy[0],cost_pair_delta_xy[1],eef_hazard_distance,eef_target_distance,episode_step_index"
  export SE_VLA_RELATION_MEMORY_AUTHORITY_GAIN=1.0
  unset SE_VLA_RELATION_MEMORY_ABSTAIN_DISTANCE || true
  unset SE_VLA_RELATION_MEMORY_RISK_GATE || true
  CK=$(ls -d $LR/v7_l1t2/newkey-h5/round_*/settled_checkpoint.json 2>/dev/null | tail -1)
  [ -n "$CK" ] || { echo "no settled checkpoint"; exit 1; }
  echo "=== v8_clean memory off=$OFF ckpt=$CK on $(hostname) ==="
  PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
    roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
    --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
    --replan-steps 1 --trials 1 --arm pruned_memory --checkpoint "$CK" \
    --port-base $PORT --array-task-id 1 --server-attempts 3
else
  echo "=== v8_clean base off=$OFF on $(hostname) ==="
  PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
    roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
    --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
    --replan-steps 1 --trials 1 --arm base \
    --port-base $PORT --array-task-id 1 --server-attempts 3
fi
echo "CLEAN_EXIT=$? mode=$MODE off=$OFF"
