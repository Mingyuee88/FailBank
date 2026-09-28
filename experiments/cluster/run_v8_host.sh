#!/usr/bin/env bash
#$ -N v8_host
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -pe smp 2
#$ -t 1-24
#$ -tc 8
#$ -o ${WORK_ROOT}/lora_dagger/v8_host/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v8_host/batch/j.$JOB_ID.$TASK_ID.out
#
# v8_host -- is policy-induced CC stable across GPU models?
#
# Three factors are now known to move CC while leaving SR untouched:
#   seed      47 offsets x 3 seeds: SR identical 47/47, polCC differs on 10/47,
#             whole-task mean 57.28 / 60.19 / 61.72  (spread 7.4%)
#   harness   pi05_stage2/run.py vs pi05_static_sweep/run.py: SR identical on the
#             5 recheck offsets, polCC differs on 2 of them (278 vs 346, 226 vs 280)
#   GPU       off3 base: 1.0 on SMALL_GPU_HOST vs 4.0/6.0 on the a6k/l40s pool
#
# SGE scatters array tasks across HOST_{D,E} (A6000) and HOST_{A,B}
# (L40S) arbitrarily, so if CC depends on GPU model then every cross-cell CC
# comparison in v8_auth carries an uncontrolled hardware term of unknown size.
# This job runs the SAME 12 offsets, same seed, same arm, pinned to one A6000
# node and one L40S node.  SR must agree offset-for-offset; the question is CC.
#
# If CC differs by host, every CC claim has to be made within a single pinned
# node (or replicated across both and reported with the spread).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

# 6 base-success (CC small, sensitive) + 6 base-failure (CC large)
OFFSETS=(3 15 19 25 35 46 0 26 29 36 42 43)
NOFF=12
i=$((SGE_TASK_ID - 1))
OFF=${OFFSETS[$((i % NOFF))]}
if [ $((i / NOFF)) -eq 0 ]; then TAG=a6k; else TAG=l40s; fi

OUT=$LR/v8_host/$TAG/off$OFF
mkdir -p "$OUT" $LR/v8_host/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi

# record what we actually landed on -- the whole point of the job
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true
cat "$OUT/host.txt"

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="host_${TAG}_off${OFF}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0

echo "=== v8_host $TAG off=$OFF ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((44000 + 25 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
echo "HOST_EXIT=$? tag=$TAG off=$OFF"
