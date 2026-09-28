#!/usr/bin/env bash
#$ -N v7_save
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-2
#$ -tc 2
#$ -l h_rt=06:00:00
#$ -o ${WORK_ROOT}/lora_dagger/v7_l1t2/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v7_l1t2/batch/j.$JOB_ID.$TASK_ID.out
# V7 gradient-free loop, ported to pi05 safety_static_obstacles L1 task2.
#
# Moved here because the frozen viability gate rejected the OpenVLA task3 testbed
# (6 usable offsets vs a threshold of 8) and its whole-task mean CC is 1.33. This
# task measures SR 0.745 / mean CC 78.2 -- 59x the headroom -- with a shield already
# verified at SR 1.00 / policy-induced CC 0, and a pipeline whose determinism was
# established over 141 runs, so n=1 per cell needs no repeats.
#
# Three arms, identical except the credit rule:
#   1 aegis-adaptive      writes on,  locked H=5 credit   (the preregistered rule)
#   2 aegis-frozen        writes off, checkpoint must stay byte-identical
#   3 aegis-adaptive-long writes on,  long_h25 credit      (reaches the measured
#                                                          12-20 step causal window)
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
ARMS=(aegis-save-h5 aegis-save-h25)
ARM=${ARMS[$((SGE_TASK_ID-1))]}
ROOT=${WORK_ROOT}/lora_dagger/v7_l1t2
mkdir -p "$ROOT/batch"
echo "=== v7_l1t2 arm=$ARM ==="; date +%s
PYTHONPATH=src:external/VLA-Arena:.:roundG/pi05_stage2 \
  external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/stage2_arms_param.py --manifest "$ROOT/manifests/${ARM}.json"
echo "V7L1T2_EXIT=$? arm=$ARM"; date +%s
