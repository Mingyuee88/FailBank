#!/usr/bin/env bash
#$ -N foldsnap
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-3
#$ -tc 3
#$ -o ${WORK_ROOT}/lora_dagger/phase2_sweep/batch/fold.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/phase2_sweep/batch/fold.$JOB_ID.$TASK_ID.out
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
cd ${SE_VLA_ROOT}
OFFSETS=(0 10 29)
OFF=${OFFSETS[$((SGE_TASK_ID-1))]}
echo "=== fold snapshots offset=$OFF ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/lora/fold_snapshots.py --offset "$OFF"
echo "FOLDSNAP_EXIT=$? offset=$OFF"
