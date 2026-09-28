#!/usr/bin/env bash
#$ -N RvE14Geo
#$ -cwd
#$ -l gpu_card=1
#$ -pe smp 4
#$ -j y
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/lora/review0913/e14_geom_probe.py ${WORK_ROOT}/lora_dagger/review0913/lists/e14_geom_probe.json
