#!/usr/bin/env bash
#$ -N p2_smoke
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -o ${WORK_ROOT}/lora_dagger/phase2_training/smoke/j.$JOB_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/phase2_training/smoke/j.$JOB_ID.out
# Phase2 LoRA-DAgger GPU smoke: offset_0, tiny batch/steps, loosened quiet-drift
# guards so the accept+adapter-save path is exercised. NOT a real training run.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
# JAX preallocates only 75% of GPU mem by default; give it nearly all 48GB.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
OP=external/VLA-Arena/vla_arena/models/openpi
OUT=${WORK_ROOT}/lora_dagger/phase2_training/smoke_out
rm -rf "$OUT"
mkdir -p ${WORK_ROOT}/lora_dagger/phase2_training/smoke
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  $OP/scripts/train_phase2_lora.py \
  --offset 0 \
  --smoke-steps 10 \
  --batch-size 2 \
  --validation-interval 5 \
  --quiet-weight 0.0 \
  --quiet-flow-loss-ratio-limit 100.0 \
  --quiet-action-drift-limit 100.0 \
  --output-root "$OUT"
echo "SMOKE_EXIT=$?"
echo "=== adapter dir ==="
find "$OUT" -maxdepth 3 | head -40
