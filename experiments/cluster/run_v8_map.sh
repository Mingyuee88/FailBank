#!/usr/bin/env bash
#$ -N v8_map
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(HOST_D*|HOST_E*)
#$ -pe smp 2
#$ -t 1-700
#$ -tc 6
#$ -o ${WORK_ROOT}/lora_dagger/v8_map/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v8_map/batch/j.$JOB_ID.$TASK_ID.out
#
# v8_map -- base capability census over the 14 unmapped safety_static_obstacles
# arenas (3 levels x 5 tasks, minus L1t2 which is already mapped in
# phase0_base_scan).  Purpose: L1t2 is exhausted -- it has only 10 base-failure
# offsets and all 10 have already been consumed as adaptation/probe sets, so the
# fix rate can never be measured cleanly there again.  We need a second arena
# with (a) enough base failures to hold out a real test set and (b) enough
# policy-induced cost for a safety method to have anything to remove.
#
# Deliberately uses roundG/pi05_stage2/run.py --arm base, i.e. the SAME harness
# and the SAME shield-off env as the deployment arms, so base and method numbers
# come from one code path.  (phase0_base_scan used pi05_static_sweep/run.py; that
# cross-harness comparison is a confound we are not repeating.)
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

# 14 arenas as "level:task_id"; L1t2 excluded (already mapped).
ARENAS=(0:0 0:1 0:2 0:3 0:4 1:0 1:1 1:3 1:4 2:0 2:1 2:2 2:3 2:4)
NOFF=50
i=$((SGE_TASK_ID - 1))
A=${ARENAS[$((i / NOFF))]}
LVL=${A%%:*}
TID=${A##*:}
OFF=$((i % NOFF))

OUT=$LR/v8_map/L${LVL}t${TID}/off${OFF}
mkdir -p "$OUT" $LR/v8_map/batch

# already done -> skip (makes the array idempotent on resubmit)
if [ -s "$OUT/result.json" ]; then echo "SKIP existing $OUT"; exit 0; fi

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="base_L${LVL}t${TID}_off${OFF}"
# run.py requires these three even when the shield is off
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
# pure base: shield off, no memory, no writes, no credit
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0

echo "=== base L${LVL} t${TID} off${OFF} (task ${SGE_TASK_ID}) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level "$LVL" --task-id "$TID" --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((20000 + 20 * (SGE_TASK_ID % 900))) --array-task-id 1 --server-attempts 3
echo "BASE_EXIT=$? L=$LVL T=$TID off=$OFF"
