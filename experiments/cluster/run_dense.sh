#!/usr/bin/env bash
#$ -N dense
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-2
#$ -tc 2
#$ -o ${WORK_ROOT}/lora_dagger/phase2_dense/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/phase2_dense/batch/j.$JOB_ID.$TASK_ID.out
# Dense re-sweep of folds 10 and 29, the two trajectories where no successful
# checkpoint was found (2 and 7 samples respectively). Codex: a finite all-failure
# trajectory does not establish unlearnability; the cheap discriminator is denser
# sampling, and for fold 10 also a wider step range.
#   validation interval 50 -> 25   (double the resolution)
#   --save-all: keep guard-REJECTED checkpoints too. Fold 10 saved only 2 of 8
#   validations because the quiet-drift guard rejected the rest, so a winner could
#   have existed and simply never been written. Rejected checkpoints are NOT
#   deployable under the safety protocol and are reported separately.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
OFFSETS=(10 29)
OFF=${OFFSETS[$((SGE_TASK_ID-1))]}
ROOT=${WORK_ROOT}/lora_dagger/phase2_dense
mkdir -p "$ROOT/batch"
echo "=== dense sweep fold offset=$OFF ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/lora/train_sweep.py --offset "$OFF" \
  --sweep-root "$ROOT" --validation-interval 25 --save-all
echo "DENSE_EXIT=$? offset=$OFF"
