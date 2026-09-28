#!/usr/bin/env bash
#$ -N p2_sweep
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-3
#$ -tc 3
#$ -o ${WORK_ROOT}/lora_dagger/phase2_sweep/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/phase2_sweep/batch/j.$JOB_ID.$TASK_ID.out
# Within-fold checkpoint sweep. Three folds chosen to span the observed range:
#   off26 lost with the LARGEST loss improvement (-69.3%)
#   off38 lost with a large improvement (-47.0%)
#   off42 WON and trained the full 400 steps (-26.2%)
# Saves an adapter at every validation so the loss-vs-outcome ordering can be
# measured within a single training trajectory instead of across folds.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
OFFSETS=(26 38 42)
OFF=${OFFSETS[$((SGE_TASK_ID-1))]}
ROOT=${WORK_ROOT}/lora_dagger/phase2_sweep
mkdir -p "$ROOT/batch"
echo "=== sweep fold offset=$OFF ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/lora/train_sweep.py --offset "$OFF" \
  --sweep-root "$ROOT" --validation-interval 50
echo "SWEEP_EXIT=$? offset=$OFF"
