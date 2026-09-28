#!/usr/bin/env bash
#$ -N Tau02Re
#$ -cwd
#$ -l gpu_card=1
#$ -pe smp 4
#$ -j y
#
# Rerun the tau=0.2 arm only, with the VALIDATION_POINT diagnostic in place.
#
# The first attempt was rejected by the quiet-drift guard but ran before train.py printed the
# per-validation numbers, so there is no record of WHICH guard it failed. That matters: on
# pi0 the failure turned out to be quiet_flow_ratio 1.477 (limit 1.1) with drift a healthy
# 0.0148 -- a genuine regression, not a mis-scaled threshold. The tau=0 arm passed cleanly
# (flow 1.052, drift 0.0065), so if tau=0.2 fails on flow too, lookahead is destabilising
# training rather than merely changing the teacher labels.
#
# Records and holdout split already exist from the first run; this only retrains.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
HO=$LR/se_tau/TAU02_ho
F=$HO/s1s2_records/derived/folds/offset_0/train.jsonl
[ -s "$F" ] || { echo "TAU02RE_ABORT: no record set at $F"; exit 1; }
echo "=== tau=0.2 retrain, rows=$(wc -l < "$F") on $(hostname) ==="
rm -rf "$HO/adapter" "$HO/ckpt" 2>/dev/null || true
RAW_ROOT=$LR/dcol_L2t0_p_tau02/records DYN_ROOT=$HO ARM_NAME=TAU02_HO \
  bash roundG/pi05_stage2/lora/run_dyn_train.sh
echo "TAU02_RETRY_DONE"
