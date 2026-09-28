#!/usr/bin/env bash
#$ -N RvE14Fin
#$ -cwd
#$ -pe smp 1
#$ -j y
set -euo pipefail
python3 ${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913/e14_final_md.py
