#!/bin/bash
#$ -N DynL1Tr
#$ -cwd
#$ -q gpu@HOST_A
#$ -l gpu_card=1
#$ -pe smp 4
#$ -j y
# Distil on safety_dynamic_obstacles L1 -- the suite the motivation figure's dynamic
# column is evaluated on.
#
# WHY. The dynamic column previously used nocurr/R2_q00, distilled on safety_STATIC_
# obstacles: zero-shot cross-category transfer, so its lower SR said nothing about the
# method on dynamic scenes. The pre-existing dynamic checkpoint (se_dyn_L2t0) is L2-tid0,
# also a mismatch. This trains on L1 tid0+tid1 directly.
#
# tau=0.0 at collection: lookahead extrapolation helps only where the hazard travels far
# (L2t0 moves 9x its radius) and HURTS in quasi-static scenes (L1t3 travelled 0.0059 and
# SR fell 72.2 -> 50.0). L1 hazard motion is unmeasured, so no extrapolation is the safe
# default.
#
# run_dyn_train.sh trains from $DYN/s1s2_records and only rebuilds it from $RAW when that
# path is ABSENT. So the merged two-task bank is written there directly; there is no
# override env var (an earlier draft invented one that the trainer does not read).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
cd ${SE_VLA_ROOT}
L=${WORK_ROOT}/lora_dagger
V=external/VLA-Arena/envs/openpi/.venv/bin/python
export PYTHONPATH=src:external/VLA-Arena:.
DYN=$L/se_dyn_L1

for t in 0 1; do
  R=$L/dcol_L1t$t/records
  [ -d "$R" ] || { echo "DL1_ABORT: missing $R"; exit 1; }
  $V roundG/pi05_stage2/lora/build_derived_c1.py --records-root "$R"
  for MODE in s1 s1s2; do
    $V roundG/pi05_stage2/lora/build_round_records.py \
      --src-root "$R" --dst-root "$L/se_dyn_L1t$t/${MODE}_records" --mode "$MODE"
  done
  echo "L1t${t}_ROWS=$(wc -l < $L/se_dyn_L1t$t/s1s2_records/derived/folds/offset_0/train.jsonl)"
done

# merge the two tasks straight into the path the trainer will read
$V roundG/pi05_stage2/lora/merge_bank.py \
  --r1 "$L/se_dyn_L1t0/s1s2_records" --r2 "$L/se_dyn_L1t1/s1s2_records" \
  --out "$DYN/s1s2_records" --fold offset_0 --tag1 t0 --tag2 t1
echo "BANK_DYN_L1_ROWS=$(wc -l < $DYN/s1s2_records/derived/folds/offset_0/train.jsonl)"

RAW_ROOT=$L/dcol_L1t0/records \
DYN_ROOT=$DYN \
ARM_NAME=DYN_L1 \
BASE_CK=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned \
LORA_CFG=pi05_vla_arena_low_mem_finetune \
QUIET_W=0.2 STEPS=800 \
  bash roundG/pi05_stage2/lora/run_dyn_train.sh
CK=$DYN/ckpt/DYN_L1/offset_0
[ -d "$CK/params" ] || { echo "DL1_ABORT: no params at $CK"; exit 1; }
$V roundG/pi05_stage2/lora/verify_family.py "$CK/params" pi05
echo "DYN_L1_TRAIN_DONE ck=$CK"
