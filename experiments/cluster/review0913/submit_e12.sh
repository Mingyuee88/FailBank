#!/usr/bin/env bash
# Review 2026-09-15, E12: held-out flow ratio vs task outcome (dose-response for tau_loss).
# Main recipe (se_r2 bank, bank_r2 blobs, lambda_q 0, data seed 1) trained for 400/1600/2400/4800
# steps with the guard limits raised to 1e9; 800 steps is the existing nocurr_s1 (ncs1).
set -euo pipefail
S=${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913
LR=${WORK_ROOT}/lora_dagger
O=$LR/review0913
J=${SE_VLA_ROOT}/joblogs
TRAINQ="gpu@HOST_A,gpu@HOST_B,gpu@HOST_C,gpu@HOST_D"
IDS=$O/lists/job_ids.txt
BASE=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
CFG=pi05_vla_arena_low_mem_finetune
ROOT=$LR/se_r2/s1s2_records
BLOBS=$LR/bank_r2/blobs
cd ${SE_VLA_ROOT}
sub() { local name=$1; shift; local out; out=$(qsub -terse "$@"); local jid=${out%%.*}; echo "$name $jid" | tee -a "$IDS" >&2; echo "$jid"; }
STEPS=(400 1600 2400 4800)
: > $O/lists/e12_train.txt; : > $O/lists/e12_fold.txt; : > $O/lists/cleanup_e12.txt
for n in "${STEPS[@]}"; do
  echo "s$n $ROOT $BLOBS $O/e12/adapter/s$n 0.0 1 $n $CFG $BASE - 1" >> $O/lists/e12_train.txt
  echo "$O/e12/adapter/s$n $O/e12/ckpt/s$n $BASE" >> $O/lists/e12_fold.txt
  echo "$O/e12/ckpt/s$n" >> $O/lists/cleanup_e12.txt
done
T=$(sub E12_train -N RvE12Tr -q "$TRAINQ" -t 1-4 -tc 4 -o $J -v LIST=$O/lists/e12_train.txt $S/train_noguard.sh)
F=$(sub E12_fold -N RvE12Fold -q "$TRAINQ" -t 1-4 -tc 4 -o $J -hold_jid $T -v LIST=$O/lists/e12_fold.txt $S/fold.sh)
EV=(); i=0
for n in "${STEPS[@]}"; do
  CK=$O/e12/ckpt/s$n/offset_0
  EV+=($(sub E12_apple_s$n -N RvE12Ap -q gpu@HOST_A -t 1-150 -tc 2 -o $J -hold_jid $F \
    -v ARM=s$n,CK=$CK,SUITE=safety_static_obstacles,LEVEL=2,TID=0,NOFF=50,REPS=3,OUTROOT=$O/e12/l2_apple,EXPECT_HOST=HOST_A,PORTBASE=$((40000 + 3100 * i)),SHIELD=observe,EXPORT_SUITE=1 $S/eval.sh))
  EV+=($(sub E12_t4_s$n -N RvE12L1 -q gpu@HOST_B -t 1-100 -tc 2 -o $J -hold_jid $F \
    -v ARM=s$n,CK=$CK,SUITE=safety_static_obstacles,LEVEL=1,TID=4,NOFF=50,REPS=2,OUTROOT=$O/e12/l1_t4,EXPECT_HOST=HOST_B,PORTBASE=$((40000 + 2100 * i)),SHIELD=observe,EXPORT_SUITE=1 $S/eval.sh))
  i=$((i + 1))
done
HOLD=$(IFS=,; echo "${EV[*]}")
sub CLEAN_E12 -N RvClean -q long -o $J -hold_jid "$HOLD" -v LIST=$O/lists/cleanup_e12.txt $S/cleanup_ckpt.sh >/dev/null
sub B3_guard_sensitivity -q long -o $J $S/b3_job.sh >/dev/null
echo "E12_B3_SUBMITTED"
