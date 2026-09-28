#!/usr/bin/env bash
#$ -N RvB4
#$ -cwd
#$ -pe smp 1
#$ -j y
set -euo pipefail
python3 ${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913/b4_label_sensitivity.py
