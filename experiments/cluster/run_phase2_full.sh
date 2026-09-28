#!/usr/bin/env bash
#$ -N p2_lora
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-9
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/phase2_training/full/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/phase2_training/full/j.$JOB_ID.$TASK_ID.out
# Phase2 LoRA-DAgger full 9-fold LOSO training. STRICT LoRA-only (~50M params,
# ~180MB adapter/fold). Each fold inits fresh from the finetuned base, trains 400
# steps (cosine 3e-5->3e-6, warmup 20), and saves adapter-only ONLY if both
# quiet-drift guards pass. Single 48GB GPU, batch 2 (proven to fit), FSDP=1.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
OP=external/VLA-Arena/vla_arena/models/openpi
OFFSETS=(0 5 10 26 27 29 38 42 43)
OFFSET=${OFFSETS[$((SGE_TASK_ID-1))]}
OUT=${WORK_ROOT}/lora_dagger/phase2_training/folds
mkdir -p "$OUT" ${WORK_ROOT}/lora_dagger/phase2_training/full
echo "=== Phase2 fold offset=$OFFSET (task $SGE_TASK_ID) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  $OP/scripts/train_phase2_lora.py \
  --offset "$OFFSET" \
  --batch-size 2 \
  --quiet-weight 0.0 \
  --output-root "$OUT"
echo "FOLD_EXIT=$? offset=$OFFSET"
