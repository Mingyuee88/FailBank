#!/usr/bin/env bash
#$ -N TauTrain
#$ -cwd
#$ -q gpu@HOST_B
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-3
#$ -tc 2
#$ -j y
#
# Train the two lookahead arms from the PINNED collections (both a6k-002).
#
# Why both arms are retrained rather than reusing DYN_L2t0_HO as the tau=0 baseline: that
# adapter came from a collection spread over a6k-001 + l40s-004. GPU model alone moves polCC
# 17.5%, so pairing it against a freshly pinned tau=0.2 arm would confound the contrast with
# exactly the artifact that already invalidated one round of this experiment.
#
# Only the holdout arms are trained (offsets 0-31, evaluated on 32-49). The full-offset arm
# would be in-sample -- collection walked 0..47 and evaluation walks 0..49 -- so it measures
# memorisation, not the transfer this comparison is about.
#
# Recipe untouched: no curriculum staging, quiet_weight 0, single 800-step phase.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python

case "$SGE_TASK_ID" in
  1) TAG=L2t0_p_tau00; ARM=TAU00 ;;
  2) TAG=L2t0_p_tau02; ARM=TAU02 ;;
  3) TAG=L2t0_p_tau01; ARM=TAU01 ;;
  *) echo "no arm"; exit 0 ;;
esac
RAW=$LR/dcol_$TAG/records
FULL=$LR/se_tau/$ARM
HO=$LR/se_tau/${ARM}_ho

n=$(ls $LR/dcol_$TAG/off*/result.json 2>/dev/null | wc -l)
[ "$n" -ge 47 ] || { echo "TAU_ABORT: $TAG only $n/47 cells collected"; exit 1; }
h=$(cat $LR/dcol_$TAG/off*/host.txt 2>/dev/null | sed 's/\..*//' | sort -u | wc -l)
[ "$h" -eq 1 ] || { echo "TAU_ABORT: $TAG spans $h hosts, not pinned"; exit 1; }
echo "=== $ARM from $TAG: $n cells on 1 host ==="

echo "--- 1. annotate + build record set"
if [ ! -s "$FULL/s1s2_records/derived/folds/offset_0/train.jsonl" ]; then
  PYTHONPATH=src:external/VLA-Arena:. $VENV roundG/pi05_stage2/lora/build_derived_c1.py \
    --records-root "$RAW"
  PYTHONPATH=src:external/VLA-Arena:. $VENV roundG/pi05_stage2/lora/build_round_records.py \
    --src-root "$RAW" --dst-root "$FULL/s1s2_records" --mode s1s2
fi
[ -s "$FULL/s1s2_records/derived/folds/offset_0/train.jsonl" ] || { echo "TAU_ABORT: no records"; exit 1; }

echo "--- 2. split holdout (train on offsets 0-31)"
if [ ! -s "$HO/s1s2_records/derived/folds/offset_0/train.jsonl" ]; then
  PYTHONPATH=src:external/VLA-Arena:. $VENV roundG/pi05_stage2/lora/make_holdout.py \
    --src "$FULL/s1s2_records" --dst "$HO/s1s2_records" --train-max 32
fi
ROWS=$(wc -l < "$HO/s1s2_records/derived/folds/offset_0/train.jsonl")
echo "    holdout rows=$ROWS"
[ "$ROWS" -ge 200 ] || { echo "TAU_ABORT: only $ROWS rows"; exit 1; }

echo "--- 3. train"
RAW_ROOT=$RAW DYN_ROOT=$HO ARM_NAME=${ARM}_HO \
  bash roundG/pi05_stage2/lora/run_dyn_train.sh
echo "TAU_TRAIN_DONE arm=$ARM tag=$TAG rows=$ROWS"
