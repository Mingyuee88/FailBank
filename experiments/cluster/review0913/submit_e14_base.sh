#!/usr/bin/env bash
# E14: complete base arms for the aligned two-backbone table on safety_hazard_avoidance.
#  pi0.5: remaining conditions (L1 r1 on HOST_E, L2 r1/r2 on HOST_D; stage-1 r0 cells are reused via SKIP)
#  pi0  : L1 50x1 on HOST_A, L2 50x3 on HOST_C
set -euo pipefail
S=${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913
O=${WORK_ROOT}/lora_dagger/review0913
J=${SE_VLA_ROOT}/joblogs
IDS=$O/lists/job_ids.txt
SU=safety_hazard_avoidance
cd ${SE_VLA_ROOT}
sub() { local name=$1; shift; local out; out=$(qsub -terse "$@"); echo "$name ${out%%.*}" | tee -a $IDS; }
for t in 0 1 2 3 4; do
  sub E14_base05_l1t${t}_r1 -N RvE14b -q gpu@HOST_E -t 2-100:2 -tc 2 -o $J \
    -v ARM=base,CK=-,SUITE=$SU,LEVEL=1,TID=$t,NOFF=50,REPS=2,OUTROOT=$O/e14/l1_t$t,EXPECT_HOST=HOST_E,PORTBASE=$((20000 + 2100 * t)),SHIELD=observe,EXPORT_SUITE=1 $S/eval.sh
  sub E14_base05_l2t${t}_r12 -N RvE14b -q gpu@HOST_D -t 1-150 -tc 2 -o $J \
    -v ARM=base,CK=-,SUITE=$SU,LEVEL=2,TID=$t,NOFF=50,REPS=3,OUTROOT=$O/e14/l2_t$t,EXPECT_HOST=HOST_D,PORTBASE=$((36000 + 3100 * t)),SHIELD=observe,EXPORT_SUITE=1 $S/eval.sh
  sub E14_base0_l1t$t -N RvE14p -q gpu@HOST_A -t 1-50 -tc 2 -o $J \
    -v ARM=base,CK=-,SUITE=$SU,LEVEL=1,TID=$t,NOFF=50,REPS=1,OUTROOT=$O/e14/pi0_l1_t$t,EXPECT_HOST=HOST_A,PORTBASE=$((20000 + 1000 * t)),SHIELD=observe,EXPORT_SUITE=1,PI0=1 $S/eval.sh
  sub E14_base0_l2t$t -N RvE14p -q gpu@HOST_C -t 1-150 -tc 2 -o $J \
    -v ARM=base,CK=-,SUITE=$SU,LEVEL=2,TID=$t,NOFF=50,REPS=3,OUTROOT=$O/e14/pi0_l2_t$t,EXPECT_HOST=HOST_C,PORTBASE=$((20000 + 3100 * t)),SHIELD=observe,EXPORT_SUITE=1,PI0=1 $S/eval.sh
done
echo E14_BASE_SUBMITTED
