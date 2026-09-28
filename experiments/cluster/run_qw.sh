#!/usr/bin/env bash
#$ -N p2_qw
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-27
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/phase2_qw/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/phase2_qw/batch/j.$JOB_ID.$TASK_ID.out
#
# STAGE A, shot 1 -- quiet_weight sweep.
#
# Verified 2026-08-08: quiet_weight is a PER-RECORD SAMPLE WEIGHT in
# DerivedRecordDataset (derived_lora_data.py:139), not a regulariser. Every Phase2
# run so far used --quiet-weight 0.0, so 1964 of each fold's 2830 training records
# (69%) were multiplied by zero. The objective fitted the shield-corrected
# triggered steps ONLY, with no term asking the policy to keep its existing
# behaviour anywhere.
#
# Phase3b measured the consequence: deployed adapters break ~7% of the offsets
# where base already succeeded (retention 78-94% per fold), and the damage
# concentrates -- offset 17 was broken by 5 of 7 adapters.
#
# This sweep asks whether turning that 69% back on trades fix-rate for retention,
# and at what rate. Nothing else changes: same trainer, same data, same guards.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
OP=external/VLA-Arena/vla_arena/models/openpi
OFFSETS=(0 5 10 26 27 29 38 42 43)
WEIGHTS=(0.1 0.3 1.0)
i=$((SGE_TASK_ID-1))
OFFSET=${OFFSETS[$((i%9))]}
QW=${WEIGHTS[$((i/9))]}
TAG=$(echo "$QW" | tr -d .)
OUT=${WORK_ROOT}/lora_dagger/phase2_qw/qw${TAG}
mkdir -p "$OUT" ${WORK_ROOT}/lora_dagger/phase2_qw/batch
echo "=== quiet_weight=$QW fold offset=$OFFSET (task $SGE_TASK_ID) ==="
date +%s
# train_fold raises if no validation checkpoint passes both quiet-drift guards.
# At higher quiet_weight that should get EASIER, so a failure here is itself a
# result worth seeing rather than something to swallow.
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  $OP/scripts/train_phase2_lora.py \
  --offset "$OFFSET" \
  --batch-size 2 \
  --quiet-weight "$QW" \
  --output-root "$OUT" || echo "QW_TRAIN_FAILED qw=$QW offset=$OFFSET"
echo "QW_EXIT=$? qw=$QW offset=$OFFSET"
date +%s
