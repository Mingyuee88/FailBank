#!/usr/bin/env bash
#$ -N p2_rep
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-45
#$ -tc 3
#$ -o ${WORK_ROOT}/lora_dagger/phase2_repeat/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/phase2_repeat/batch/j.$JOB_ID.$TASK_ID.out
#
# TRAINING-RUN VARIANCE. Verified 2026-08-08 by direct weight comparison: re-running
# the identical fold/config produces DIFFERENT adapters -- 0 of 20 tensors identical,
# max relative difference 0.4%-1.8% per fold. Training is nondeterministic (unlike
# evaluation, which is bit-deterministic; the deterministic XLA flags are applied
# only to the policy-server subprocess, never to training).
#
# Consequence: Phase3a's SR 4/9 is ONE DRAW from a distribution over training runs,
# not a property of the method. This job measures that distribution.
#
# Design: the STOCK Phase2 trainer, unchanged (same script, same args as
# run_phase2_full.sh) -- each repeat is a fair draw of exactly what Phase2 would
# have deployed, early stopping and guard included. 9 folds x 5 repeats.
# Adapter-only output is ~180 MB/run -> 45 x 180 MB = 8.1 GB total.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
OP=external/VLA-Arena/vla_arena/models/openpi
OFFSETS=(0 5 10 26 27 29 38 42 43)
i=$((SGE_TASK_ID-1))
OFFSET=${OFFSETS[$((i%9))]}
REP=$((i/9+1))
OUT=${WORK_ROOT}/lora_dagger/phase2_repeat/rep${REP}
mkdir -p "$OUT" ${WORK_ROOT}/lora_dagger/phase2_repeat/batch
echo "=== repeat rep=$REP fold offset=$OFFSET (task $SGE_TASK_ID) ==="
date +%s
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  $OP/scripts/train_phase2_lora.py \
  --offset "$OFFSET" \
  --batch-size 2 \
  --quiet-weight 0.0 \
  --output-root "$OUT"
echo "REPEAT_EXIT=$? rep=$REP offset=$OFFSET"
date +%s
