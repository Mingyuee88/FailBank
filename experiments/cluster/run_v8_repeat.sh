#!/usr/bin/env bash
#$ -N v8_repeat
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 2
#$ -t 1-36
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v8_repeat/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v8_repeat/batch/j.$JOB_ID.$TASK_ID.out
#
# v8_repeat -- IS THE PIPELINE REPEATABLE ONCE THE NODE IS PINNED?
#
# This gates every further experiment.  v8_host showed that with everything else
# held identical (same offset, same seed, same arm, same code), swapping the GPU
# model moves policy-induced CC on 11 of 12 offsets, mean 146.5 (A6000) vs 172.2
# (L40S) = +17.5%, and flips SR on 1 of 12 (off35: success/pCC 7 on A6000,
# failure/pCC 386 on L40S).  That hardware term is 5x the entire measured method
# effect (-3.7%), and SGE scatters array tasks across both models, so every
# cross-cell comparison made so far carries it.
#
# The question this job answers: with the node PINNED to one host, is a rerun of
# an identical configuration bit-identical?
#
#   If YES -> the fix is purely procedural (pin the node) and paired n=1
#             comparisons become valid again; the whole v8_auth sweep can be
#             re-run cleanly on one node.
#   If NO  -> there is genuine run-to-run non-determinism inside the pipeline
#             (JAX autotuning, non-deterministic reductions, thread scheduling),
#             every CC number needs replication, and the required n has to be
#             estimated from the spread measured here.
#
# 12 offsets x 3 identical repeats, everything else frozen.  Repeats are written
# to separate directories so nothing is overwritten.  6 base-success offsets
# (small CC, sensitive to perturbation) + 6 base-failure offsets (large CC).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

OFFSETS=(3 15 19 25 35 46 0 26 29 36 42 43)
NOFF=12
i=$((SGE_TASK_ID - 1))
OFF=${OFFSETS[$((i % NOFF))]}
REP=$((i / NOFF + 1))

OUT=$LR/v8_repeat/rep$REP/off$OFF
mkdir -p "$OUT" $LR/v8_repeat/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="repeat_r${REP}_off${OFF}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0

echo "=== v8_repeat rep=$REP off=$OFF on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((36000 + 25 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
echo "REPEAT_EXIT=$? rep=$REP off=$OFF"
