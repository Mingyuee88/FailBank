#!/usr/bin/env bash
#$ -N v8_seedpin
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_B
#$ -pe smp 2
#$ -t 1-36
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v8_seedpin/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v8_seedpin/batch/j.$JOB_ID.$TASK_ID.out
#
# v8_seedpin -- is --seed a genuine no-op once the node is pinned?
#
# v8_repeat settled that with the node pinned, identical reruns are bit-identical
# (11/11 offsets, SR and polCC both, 0.0% spread).  So the pipeline itself is
# deterministic and every wobble seen so far is an environment term.
#
# That reopens the earlier "seed variance" result.  phase0_base_scan measured
# 47 offsets x 3 seeds and found polCC differing on 10/47 with a 7.4% spread in
# the whole-task mean -- but that scan ran under `-l h=(A6K_HOSTS*|L40S_HOSTS*)`, so the
# three seeds of a given offset almost certainly landed on DIFFERENT GPU models.
# The "seed effect" and the hardware effect are completely confounded there.
#
# This job separates them: same 12 offsets, three seeds, all pinned to ONE host.
#
#   seeds identical -> --seed really is a no-op (as the 08-06 handoff claimed),
#                      ALL observed variance was hardware, and replication buys
#                      nothing: the only way to gain statistical power is more
#                      OFFSETS.  That decides the merged-arena design.
#   seeds differ    -> seed is a real factor after all and every cell needs >=3.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

OFFSETS=(3 15 19 25 35 46 0 26 29 36 42 43)
SEEDS=(7 11 13)
NOFF=12
i=$((SGE_TASK_ID - 1))
OFF=${OFFSETS[$((i % NOFF))]}
SD=${SEEDS[$((i / NOFF))]}

OUT=$LR/v8_seedpin/seed$SD/off$OFF
mkdir -p "$OUT" $LR/v8_seedpin/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=$SD SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="seedpin_s${SD}_off${OFF}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0

echo "=== v8_seedpin seed=$SD off=$OFF on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed "$SD" \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((28000 + 25 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
echo "SEEDPIN_EXIT=$? seed=$SD off=$OFF"
