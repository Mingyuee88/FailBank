#!/usr/bin/env bash
# Review 2026-09-14: last-step diagnostic for the guard-rejected E5 il_r1 and E6 r2only.
# train (guard limits 1e9) -> fold -> same three evaluations as the E5/E6 arms -> cleanup.
set -euo pipefail
S=${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913
LR=${WORK_ROOT}/lora_dagger
O=$LR/review0913
J=${SE_VLA_ROOT}/joblogs
TRAINQ="gpu@HOST_A,gpu@HOST_B,gpu@HOST_C,gpu@HOST_D"
IDS=$O/lists/job_ids.txt
STATIC=safety_static_obstacles
BASE=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
CFG=pi05_vla_arena_low_mem_finetune
cd ${SE_VLA_ROOT}

sub() {
  local name=$1; shift
  local out; out=$(qsub -terse "$@")
  local jid=${out%%.*}
  echo "$name $jid" | tee -a "$IDS" >&2
  echo "$jid"
}

# Same records, blobs, lambda_q, seed, steps and config as the rejected runs (lists/e56_train.txt).
cat > $O/lists/diag_train.txt <<EOF
il_r1_diag $O/e5/il_r1_records $O/e5/il_r1_records/blobs $O/e5/adapter/il_r1_diag 0.0 1 800 $CFG $BASE - 1
r2only_diag $O/e6/r2only_records $O/e6/r2only_records/blobs $O/e6/adapter/r2only_diag 0.0 1 800 $CFG $BASE - 1
EOF
cat > $O/lists/diag_fold.txt <<EOF
$O/e5/adapter/il_r1_diag $O/e5/ckpt/il_r1_diag $BASE
$O/e6/adapter/r2only_diag $O/e6/ckpt/r2only_diag $BASE
EOF
printf "%s\n" $O/e5/ckpt/il_r1_diag $O/e6/ckpt/r2only_diag > $O/lists/cleanup_diag.txt
# The records roots must match the ones the rejected runs used.
grep -q "il_r1 $O/e5/il_r1_records " $O/lists/e56_train.txt
grep -q "r2only $O/e6/r2only_records " $O/lists/e56_train.txt

T=$(sub DIAG_train -N RvTrNG -q "$TRAINQ" -t 1-2 -tc 2 -o $J -v LIST=$O/lists/diag_train.txt $S/train_noguard.sh)
F=$(sub DIAG_fold -N RvFoldNG -q "$TRAINQ" -t 1-2 -tc 2 -o $J -hold_jid $T -v LIST=$O/lists/diag_fold.txt $S/fold.sh)

EV=()
i=0
for a in il_r1_diag r2only_diag; do
  case $a in il_r1_diag) e=e5 ;; *) e=e6 ;; esac
  CK=$O/$e/ckpt/$a/offset_0
  # L2 apple 50 x 3 on l40s-004 (pairs with l2_apple base/ncs1 and the E5/E6 arms)
  EV+=($(sub DIAG_apple_$a -N RvApNG -q gpu@HOST_A -t 1-150 -tc 2 -o $J -hold_jid $F \
    -v ARM=$a,CK=$CK,SUITE=$STATIC,LEVEL=2,TID=0,NOFF=50,REPS=3,OUTROOT=$O/e56/l2_apple,EXPECT_HOST=HOST_A,PORTBASE=$((20000 + 3100 * i)),SHIELD=observe,EXPORT_SUITE=1 \
    $S/eval.sh))
  # L1 T1 on a6k-001 and T4 on l40s-005, 50 x 2 (pairs with multi_t1/t4)
  EV+=($(sub DIAG_t1_$a -N RvL1NG -q gpu@HOST_C -t 1-100 -tc 2 -o $J -hold_jid $F \
    -v ARM=$a,CK=$CK,SUITE=$STATIC,LEVEL=1,TID=1,NOFF=50,REPS=2,OUTROOT=$O/e56/l1_t1,EXPECT_HOST=HOST_C,PORTBASE=$((60000 + 2100 * i)),SHIELD=observe,EXPORT_SUITE=1 \
    $S/eval.sh))
  EV+=($(sub DIAG_t4_$a -N RvL1NG -q gpu@HOST_B -t 1-100 -tc 2 -o $J -hold_jid $F \
    -v ARM=$a,CK=$CK,SUITE=$STATIC,LEVEL=1,TID=4,NOFF=50,REPS=2,OUTROOT=$O/e56/l1_t4,EXPECT_HOST=HOST_B,PORTBASE=$((60000 + 2100 * i)),SHIELD=observe,EXPORT_SUITE=1 \
    $S/eval.sh))
  i=$((i + 1))
done
HOLD=$(IFS=,; echo "${EV[*]}")
sub CLEAN_DIAG -N RvClean -q long -o $J -hold_jid "$HOLD" -v LIST=$O/lists/cleanup_diag.txt $S/cleanup_ckpt.sh >/dev/null
echo "DIAG_SUBMITTED"
