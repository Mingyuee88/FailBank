#!/usr/bin/env bash
# E14 stage 1: pi0.5 base on safety_hazard_avoidance, condition r0 only (headroom + cost audit).
set -euo pipefail
S=${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913
O=${WORK_ROOT}/lora_dagger/review0913
J=${SE_VLA_ROOT}/joblogs
IDS=$O/lists/job_ids.txt
[ -e $O/e14 ] && { echo "E14_ABORT exists"; exit 1; }
mkdir -p $O/e14
cd ${SE_VLA_ROOT}
for t in 0 1 2 3 4; do
  out=$(qsub -terse -N RvE14b -q gpu@HOST_E -t 1-99:2 -tc 2 -o $J \
    -v ARM=base,CK=-,SUITE=safety_hazard_avoidance,LEVEL=1,TID=$t,NOFF=50,REPS=2,OUTROOT=$O/e14/l1_t$t,EXPECT_HOST=HOST_E,PORTBASE=$((20000 + 2100 * t)),SHIELD=observe,EXPORT_SUITE=1 $S/eval.sh)
  echo "E14_s1_base_l1t$t ${out%%.*}" | tee -a $IDS
  out=$(qsub -terse -N RvE14b -q gpu@HOST_D -t 1-148:3 -tc 2 -o $J \
    -v ARM=base,CK=-,SUITE=safety_hazard_avoidance,LEVEL=2,TID=$t,NOFF=50,REPS=3,OUTROOT=$O/e14/l2_t$t,EXPECT_HOST=HOST_D,PORTBASE=$((20000 + 3100 * t)),SHIELD=observe,EXPORT_SUITE=1 $S/eval.sh)
  echo "E14_s1_base_l2t$t ${out%%.*}" | tee -a $IDS
done
echo E14_S1_SUBMITTED
