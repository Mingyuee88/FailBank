#!/usr/bin/env bash
# Review 2026-09-14, E10: Level-2 T1 (lemon) and T4 (tomato), arms base + ncs1-3.
# The AEGIS arm is submitted separately once the GLM-4.5V endpoint answers.
set -euo pipefail
S=${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913
L=${SE_VLA_ROOT}/roundG/pi05_stage2/lora
LR=${WORK_ROOT}/lora_dagger
O=$LR/review0913
J=${SE_VLA_ROOT}/joblogs
TRAINQ="gpu@HOST_A,gpu@HOST_B,gpu@HOST_C,gpu@HOST_D"
IDS=$O/lists/job_ids.txt
STATIC=safety_static_obstacles
cd ${SE_VLA_ROOT}

sub() {
  local name=$1; shift
  local out; out=$(qsub -terse "$@")
  local jid=${out%%.*}
  echo "$name $jid" | tee -a "$IDS" >&2
  echo "$jid"
}

# Refold the three replicate adapters exactly as E0 did (same list lines, bit-identical fold).
grep -E "/nocurr_s[123]/adapter/R2_q00 " $O/lists/e0_fold.txt > $O/lists/e10_fold.txt
[ "$(wc -l < $O/lists/e10_fold.txt)" = 3 ] || { echo "E10_ABORT fold list"; exit 1; }
printf "%s\n" $LR/nocurr_s1/ckpt/R2_q00 $LR/nocurr_s2/ckpt/R2_q00 $LR/nocurr_s3/ckpt/R2_q00 > $O/lists/cleanup_e10.txt
F=$(sub E10_fold -N RvE10Fold -q "$TRAINQ" -t 1-3 -tc 3 -o $J -v LIST=$O/lists/e10_fold.txt $S/fold.sh)

EV=()
for spec in "1 lemon HOST_A" "4 tomato HOST_B"; do
  read -r TID TAG HOST <<< "$spec"
  OUTROOT=$O/e10/l2_$TAG
  # base: same observe-only wiring as run_l2_task.sh's base arm
  sub E10_${TAG}_base -N RvE10b -q gpu@$HOST -t 1-150 -tc 2 -o $J \
    -v ARM=base,CK=-,SUITE=$STATIC,LEVEL=2,TID=$TID,NOFF=50,REPS=3,OUTROOT=$OUTROOT,EXPECT_HOST=$HOST,PORTBASE=20000,SHIELD=observe,EXPORT_SUITE=1 \
    $S/eval.sh >/dev/null
  p=23100
  for s in 1 2 3; do
    EV+=($(sub E10_${TAG}_ncs$s -N RvE10n -q gpu@$HOST -t 1-150 -tc 2 -o $J -hold_jid $F \
      -v ARM=ncs$s,CK=$LR/nocurr_s$s/ckpt/R2_q00/offset_0,SUITE=$STATIC,LEVEL=2,TID=$TID,NOFF=50,REPS=3,OUTROOT=$OUTROOT,EXPECT_HOST=$HOST,PORTBASE=$p,SHIELD=observe,EXPORT_SUITE=1 \
      $S/eval.sh))
    p=$((p + 3100))
  done
done
HOLD=$(IFS=,; echo "${EV[*]}")
sub CLEAN_E10 -N RvClean -q long -o $J -hold_jid "$HOLD" -v LIST=$O/lists/cleanup_e10.txt $S/cleanup_ckpt.sh >/dev/null

# GLM-4.5V service for the AEGIS arm (same launcher and weights as every earlier AEGIS cell).
if qstat -u USER 2>/dev/null | grep -q glm45v; then
  echo "GLM job already queued"
else
  cd $L && sub GLM_serve_e10 $L/run_glm_serve_a6k.sh >/dev/null
fi
echo "E10_SUBMITTED"
