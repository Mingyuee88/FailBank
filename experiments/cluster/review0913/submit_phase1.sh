#!/usr/bin/env bash
# Review 2026-09-13, phase 1: every job that does not depend on the record builds.
# Appends "NAME JOB_ID" lines to $O/lists/job_ids.txt.
set -euo pipefail
S=${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913
L=${SE_VLA_ROOT}/roundG/pi05_stage2/lora
LR=${WORK_ROOT}/lora_dagger
O=$LR/review0913
J=${SE_VLA_ROOT}/joblogs
# Only l40s-004/005 and a6k-001/002 belong to GROUP_lab; other l40s hosts leave jobs in qw forever.
TRAINQ="gpu@HOST_A,gpu@HOST_B,gpu@HOST_C,gpu@HOST_D"
IDS=$O/lists/job_ids.txt
cd ${SE_VLA_ROOT}

sub() {
  local name=$1; shift
  local out; out=$(qsub -terse "$@")
  local jid=${out%%.*}
  echo "$name $jid" | tee -a "$IDS" >&2
  echo "$jid"
}
STATIC=safety_static_obstacles
DYN=safety_dynamic_obstacles

# ---------------- E0: refold main adapter + three data-order replicates ----------------
E0=$(sub E0_fold -N RvE0Fold -q "$TRAINQ" -t 1-4 -tc 4 -o $J -v LIST=$O/lists/e0_fold.txt $S/fold.sh)

# ---------------- E4: the labelling teacher executed in the loop, static L2 ----------------
# Mirrors run_l2_task.sh (50 offsets x 3 sampler conditions, SE_VLA_ARENA_SUITE exported),
# same host per task as the published base/AEGIS/FailBank cells.
sub E4_apple -N RvE4ap -q gpu@HOST_A -t 1-150 -tc 2 -o $J \
  -v ARM=teacherloop,CK=-,SUITE=$STATIC,LEVEL=2,TID=0,NOFF=50,REPS=3,OUTROOT=$O/e4/l2_apple,EXPECT_HOST=HOST_A,PORTBASE=60000,SHIELD=inloop_local,EXPORT_SUITE=1 \
  $S/eval.sh >/dev/null
sub E4_mango -N RvE4mg -q gpu@HOST_D -t 1-150 -tc 2 -o $J \
  -v ARM=teacherloop,CK=-,SUITE=$STATIC,LEVEL=2,TID=2,NOFF=50,REPS=3,OUTROOT=$O/e4/l2_full,EXPECT_HOST=HOST_D,PORTBASE=60000,SHIELD=inloop_local,EXPORT_SUITE=1 \
  $S/eval.sh >/dev/null
sub E4_onion -N RvE4on -q gpu@HOST_B -t 1-150 -tc 2 -o $J \
  -v ARM=teacherloop,CK=-,SUITE=$STATIC,LEVEL=2,TID=3,NOFF=50,REPS=3,OUTROOT=$O/e4/l2_onion,EXPECT_HOST=HOST_B,PORTBASE=60000,SHIELD=inloop_local,EXPORT_SUITE=1 \
  $S/eval.sh >/dev/null

# ---------------- E2: main static adapter on dynamic L1-T3 / T4 (after E0) ----------------
# Mirrors run_tau_eval_t3/t4.sh: 18 offsets, 1 condition, SE_VLA_ARENA_SUITE NOT exported,
# lookahead 0, host HOST_D, so it pairs with tau_eval_t3/t4 base and localsh cells.
k=0
for arm in nocurr ncs1 ncs2 ncs3; do
  case $arm in nocurr) d=nocurr ;; ncs1) d=nocurr_s1 ;; ncs2) d=nocurr_s2 ;; ncs3) d=nocurr_s3 ;; esac
  CK=$LR/$d/ckpt/R2_q00/offset_0
  sub E2_t3_$arm -N RvE2t3 -q gpu@HOST_D -t 1-18 -tc 2 -o $J -hold_jid $E0 \
    -v ARM=$arm,CK=$CK,SUITE=$DYN,LEVEL=1,TID=3,NOFF=18,REPS=1,OUTROOT=$O/e2/tau_eval_t3,EXPECT_HOST=HOST_D,PORTBASE=$((52000 + 400 * k)),SHIELD=observe,EXPORT_SUITE=0 \
    $S/eval.sh >/dev/null
  sub E2_t4_$arm -N RvE2t4 -q gpu@HOST_D -t 1-18 -tc 2 -o $J -hold_jid $E0 \
    -v ARM=$arm,CK=$CK,SUITE=$DYN,LEVEL=1,TID=4,NOFF=18,REPS=1,OUTROOT=$O/e2/tau_eval_t4,EXPECT_HOST=HOST_D,PORTBASE=$((55000 + 400 * k)),SHIELD=observe,EXPORT_SUITE=0 \
    $S/eval.sh >/dev/null
  k=$((k + 1))
done

# ---------------- E3: static L1 T0 / T3 with the verified launcher (after E0) ----------------
# run_multitask.sh ARMS index 4..8 = aegis nocurr ncs1 ncs2 ncs3 -> task ids 401..900.
# Existing base/r1b/r2 cells are on l40s-005 (t0) and a6k-001 (t3); SKIP logic protects them.
sub E3_t0 -N RvE3t0 -q gpu@HOST_B -t 401-900 -tc 3 -o $J -hold_jid $E0 \
  -v TID=0,TAG=t0,EXPECT_HOST=HOST_B $L/run_multitask.sh >/dev/null
sub E3_t3 -N RvE3t3 -q gpu@HOST_C -t 401-900 -tc 3 -o $J -hold_jid $E0 \
  -v TID=3,TAG=t3,EXPECT_HOST=HOST_C $L/run_multitask.sh >/dev/null

# ---------------- E7: per-step deployment wall clock, L2 apple (after E0) ----------------
CK1=$LR/nocurr_s1/ckpt/R2_q00/offset_0
sub E7_base -N RvE7b -q gpu@HOST_A -t 1-10 -tc 1 -o $J -hold_jid $E0 \
  -v ARM=base,CK=-,SUITE=$STATIC,LEVEL=2,TID=0,NOFF=10,REPS=1,OUTROOT=$O/e7/l2_apple,EXPECT_HOST=HOST_A,PORTBASE=64000,SHIELD=observe,EXPORT_SUITE=1,TIMING=1 \
  $S/eval.sh >/dev/null
sub E7_failbank -N RvE7f -q gpu@HOST_A -t 1-10 -tc 1 -o $J -hold_jid $E0 \
  -v ARM=failbank_ncs1,CK=$CK1,SUITE=$STATIC,LEVEL=2,TID=0,NOFF=10,REPS=1,OUTROOT=$O/e7/l2_apple,EXPECT_HOST=HOST_A,PORTBASE=64300,SHIELD=observe,EXPORT_SUITE=1,TIMING=1 \
  $S/eval.sh >/dev/null
sub E7_aegis -N RvE7a -q gpu@HOST_A -t 1-10 -tc 1 -o $J -hold_jid $E0 \
  -v ARM=aegis,CK=-,SUITE=$STATIC,LEVEL=2,TID=0,NOFF=10,REPS=1,OUTROOT=$O/e7/l2_apple,EXPECT_HOST=HOST_A,PORTBASE=64600,SHIELD=aegis,EXPORT_SUITE=1,TIMING=1 \
  $S/eval.sh >/dev/null

# ---------------- E8: pi0 first-K supervision, train then fold ----------------
E8T=$(sub E8_train -N RvE8Tr -q "$TRAINQ" -t 1-2 -tc 2 -o $J -v LIST=$O/lists/e8_train.txt $S/train.sh)
sub E8_fold -N RvE8Fold -q "$TRAINQ" -t 1-2 -tc 2 -o $J -hold_jid $E8T -v LIST=$O/lists/e8_fold.txt $S/fold.sh >/dev/null

# ---------------- E9: dynamic round 1 at 700 steps -> fold -> eval on dyn L2-T0 ----------------
# Mirrors run_dyn_r3.sh: 50 offsets, 1 condition, suite exported, lookahead 0, host l40s-005.
E9T=$(sub E9_train -N RvE9Tr -q "$TRAINQ" -t 1-1 -o $J -v LIST=$O/lists/e9_train.txt $S/train.sh)
E9F=$(sub E9_fold -N RvE9Fold -q "$TRAINQ" -t 1-1 -o $J -hold_jid $E9T -v LIST=$O/lists/e9_fold.txt $S/fold.sh)
sub E9_eval -N RvE9Ev -q gpu@HOST_B -t 1-50 -tc 2 -o $J -hold_jid $E9F \
  -v ARM=dyn_r1_700,CK=$O/e9/ckpt/dyn_r1_700/offset_0,SUITE=$DYN,LEVEL=2,TID=0,NOFF=50,REPS=1,OUTROOT=$O/e9/dyn_l2_t0,EXPECT_HOST=HOST_B,PORTBASE=58000,SHIELD=observe,EXPORT_SUITE=1 \
  $S/eval.sh >/dev/null

echo "PHASE1_SUBMITTED"
