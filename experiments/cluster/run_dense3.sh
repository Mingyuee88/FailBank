#!/usr/bin/env bash
#$ -N dense3
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-3
#$ -tc 3
#$ -o ${WORK_ROOT}/lora_dagger/phase2_dense3/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/phase2_dense3/batch/j.$JOB_ID.$TASK_ID.out
#
# Snapshot sweep of folds 5, 27, 43 -- the ONLY three LOSO folds never swept and
# never used to derive the step-25 rule (Codex boundary #2 contaminates 0, 10, 26,
# 29, 38, 42). Completes the sweep to all 9 folds.
#
# Note these three are already WINNERS under best-held-out-loss in Phase3a, so this
# is a prospective NON-DEGRADATION test of step-25, not a "recover lost SR" test.
# Preregistration frozen before submission: phase2_dense3/PREREG_step25.md
#
# Same knobs as the proven run_dense.sh: interval 25, --save-all so guard-rejected
# checkpoints are also written (reported separately; they are NOT deployable).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
OFFSETS=(5 27 43)
OFF=${OFFSETS[$((SGE_TASK_ID-1))]}
ROOT=${WORK_ROOT}/lora_dagger/phase2_dense3
mkdir -p "$ROOT/batch"
echo "=== dense3 sweep fold offset=$OFF ==="
date +%s
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/lora/train_sweep.py --offset "$OFF" \
  --sweep-root "$ROOT" --validation-interval 25 --save-all
echo "DENSE3_EXIT=$? offset=$OFF"
date +%s
