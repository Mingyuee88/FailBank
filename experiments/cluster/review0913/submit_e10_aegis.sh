#!/usr/bin/env bash
# Review 2026-09-14, E10 AEGIS arm for L2 T1 (lemon) and T4 (tomato), held on the GLM probe.
set -euo pipefail
S=${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913
O=${WORK_ROOT}/lora_dagger/review0913
J=${SE_VLA_ROOT}/joblogs
IDS=$O/lists/job_ids.txt
cd ${SE_VLA_ROOT}
sub() {
  local name=$1; shift
  local out; out=$(qsub -terse "$@")
  local jid=${out%%.*}
  echo "$name $jid" | tee -a "$IDS" >&2
  echo "$jid"
}
P=$(sub E10_glm_probe -q long -o $J $S/glm_probe.sh)
for spec in "1 lemon HOST_A" "4 tomato HOST_B"; do
  read -r TID TAG HOST <<< "$spec"
  sub E10_${TAG}_aegis -N RvE10a -q gpu@$HOST -t 1-150 -tc 2 -o $J -hold_jid $P \
    -v ARM=aegis,CK=-,SUITE=safety_static_obstacles,LEVEL=2,TID=$TID,NOFF=50,REPS=3,OUTROOT=$O/e10/l2_$TAG,EXPECT_HOST=$HOST,PORTBASE=32400,SHIELD=aegis,EXPORT_SUITE=1 \
    $S/eval.sh >/dev/null
done
echo "E10_AEGIS_SUBMITTED"
