#!/usr/bin/env bash
# Review 2026-09-13, phase 2: jobs that need the record builds (run_builds.sh must have
# printed BUILDS_DONE). Chains train -> fold -> eval with -hold_jid.
set -euo pipefail
S=${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913
LR=${WORK_ROOT}/lora_dagger
O=$LR/review0913
J=${SE_VLA_ROOT}/joblogs
TRAINQ="gpu@HOST_A,gpu@HOST_B,gpu@HOST_C,gpu@HOST_D"
IDS=$O/lists/job_ids.txt
STATIC=safety_static_obstacles
cd ${SE_VLA_ROOT}
grep -q BUILDS_DONE $O/lists/builds.log || { echo "PHASE2_ABORT: builds not finished"; exit 1; }

sub() {
  local name=$1; shift
  local out; out=$(qsub -terse "$@")
  local jid=${out%%.*}
  echo "$name $jid" | tee -a "$IDS" >&2
  echo "$jid"
}

# L2 apple eval helper: 50 offsets x 3 conditions on l40s-004, pairs with l2_apple base/ncs*.
apple() {  # name arm ck portbase hold outroot
  sub "$1" -N RvAp -q gpu@HOST_A -t 1-150 -tc 2 -o $J -hold_jid "$5" \
    -v ARM=$2,CK=$3,SUITE=$STATIC,LEVEL=2,TID=0,NOFF=50,REPS=3,OUTROOT=$6,EXPECT_HOST=HOST_A,PORTBASE=$4,SHIELD=observe,EXPORT_SUITE=1 \
    $S/eval.sh >/dev/null
}
# L1 T1 (a6k-001) and T4 (l40s-005): 50 offsets x 2 conditions, pairs with multi_t1/t4.
l1() {  # name arm ck tid host portbase hold outroot
  sub "$1" -N RvL1 -q gpu@$5 -t 1-100 -tc 2 -o $J -hold_jid "$7" \
    -v ARM=$2,CK=$3,SUITE=$STATIC,LEVEL=1,TID=$4,NOFF=50,REPS=2,OUTROOT=$8,EXPECT_HOST=$5,PORTBASE=$6,SHIELD=observe,EXPORT_SUITE=1 \
    $S/eval.sh >/dev/null
}

# ---------------- E1 stage 1: SFT0 / SHAM / SFTPOS, data seed 1 ----------------
E1T=$(sub E1_train_s1 -N RvE1Tr -q "$TRAINQ" -t 1-3 -tc 3 -o $J -v LIST=$O/lists/e1_train.txt $S/train.sh)
E1F=$(sub E1_fold_s1 -N RvE1Fold -q "$TRAINQ" -t 1-3 -tc 3 -o $J -hold_jid $E1T -v LIST=$O/lists/e1_fold.txt $S/fold.sh)
apple E1_apple_sft0_s1   sft0_s1   $O/e1/ckpt/sft0_s1/offset_0   30000 $E1F $O/e1/l2_apple
apple E1_apple_sham_s1   sham_s1   $O/e1/ckpt/sham_s1/offset_0   33100 $E1F $O/e1/l2_apple
apple E1_apple_sftpos_s1 sftpos_s1 $O/e1/ckpt/sftpos_s1/offset_0 36200 $E1F $O/e1/l2_apple

# ---------------- E1 stage 2: seeds 2 and 3 on L2 apple; seed 1 on L1 T1/T4 ----------------
E1T2=$(sub E1_train_s23 -N RvE1Tr2 -q "$TRAINQ" -t 1-6 -tc 3 -o $J -hold_jid $E1T -v LIST=$O/lists/e1b_train.txt $S/train.sh)
E1F2=$(sub E1_fold_s23 -N RvE1Fold2 -q "$TRAINQ" -t 1-6 -tc 3 -o $J -hold_jid $E1T2 -v LIST=$O/lists/e1b_fold.txt $S/fold.sh)
p=20000
for a in sft0_s2 sft0_s3 sham_s2 sham_s3 sftpos_s2 sftpos_s3; do
  apple E1_apple_$a $a $O/e1/ckpt/$a/offset_0 $p $E1F2 $O/e1/l2_apple
  p=$((p + 3100))
done
for a in sft0_s1 sham_s1 sftpos_s1; do :; done
q=40000
for a in sft0_s1 sham_s1 sftpos_s1; do
  l1 E1_t1_$a $a $O/e1/ckpt/$a/offset_0 1 HOST_C $((q + 10000)) $E1F $O/e1/l1_t1
  l1 E1_t4_$a $a $O/e1/ckpt/$a/offset_0 4 HOST_B $((q + 10000)) $E1F $O/e1/l1_t4
  q=$((q + 2100))
done

# ---------------- E5 + E6: bank construction ablations ----------------
E56T=$(sub E56_train -N RvE56Tr -q "$TRAINQ" -t 1-4 -tc 4 -o $J -v LIST=$O/lists/e56_train.txt $S/train.sh)
E56F=$(sub E56_fold -N RvE56Fold -q "$TRAINQ" -t 1-4 -tc 4 -o $J -hold_jid $E56T -v LIST=$O/lists/e56_fold.txt $S/fold.sh)
i=0
for a in oo_r1 il_r1 r2only accum_matched; do
  case $a in oo_r1|il_r1) e=e5 ;; *) e=e6 ;; esac
  CK=$O/$e/ckpt/$a/offset_0
  apple E56_apple_$a $a $CK $((40000 + 3100 * i)) $E56F $O/e56/l2_apple
  l1 E56_t1_$a $a $CK 1 HOST_C  $((40000 + 2100 * i)) $E56F $O/e56/l1_t1
  l1 E56_t4_$a $a $CK 4 HOST_B $((40000 + 2100 * i)) $E56F $O/e56/l1_t4
  i=$((i + 1))
done

echo "PHASE2_SUBMITTED"
