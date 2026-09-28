#!/usr/bin/env bash
#$ -N r0_ctrl
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-2
#$ -tc 2
#$ -o ${WORK_ROOT}/lora_dagger/round0_control/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/round0_control/batch/j.$JOB_ID.$TASK_ID.out
# Round 0 CONTROL: the existing self-evolution loop, LOCKED credit rule, on L2_t4.
# Task 1 = aegis-adaptive (writes on, checkpoint chains forward)
# Task 2 = aegis-frozen  (writes off, checkpoint must stay byte-identical)
# Each chain runs 5 offsets x (adapt + probe) sequentially.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=600
cd ${SE_VLA_ROOT}

ARMS=(aegis-adaptive aegis-frozen)
ARM=${ARMS[$((SGE_TASK_ID-1))]}
ROOT=${WORK_ROOT}/lora_dagger/round0_control
mkdir -p "$ROOT/batch"

echo "=== round0 control arm=$ARM ==="
PYTHONPATH=src:external/VLA-Arena:.:roundG/pi05_stage2 \
  external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/stage2_arms_param.py --manifest "$ROOT/manifests/${ARM}.json"
echo "ROUND0_EXIT=$? arm=$ARM"
