#!/usr/bin/env bash
# E13: move pi0 L2 apple/lemon evaluation (both new seeds) from HOST_B to HOST_A.
# Only runs if the host check passed, l40s-005 has no free GPU, and none of the four arrays has started.
set -euo pipefail
S=${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913
O=${WORK_ROOT}/lora_dagger/review0913
J=${SE_VLA_ROOT}/joblogs
IDS=$O/lists/job_ids.txt
OLD="1448953 1448954 1448965 1448966"
FOLD=1448945; CLEAN=1448970
python3 $S/hostcheck_compare.py | tee $O/lists/e13_hostcheck.txt | grep -q "^HOSTCHECK_PASS" || { echo "MOVE_ABORT host check not passed"; exit 1; }
free=$(qhost -F gpu_card -h HOST_B | grep -oE "gpu_card=[0-9.]+" | head -1 | cut -d= -f2)
awk "BEGIN{exit !($free < 1)}" || { echo "MOVE_SKIP l40s-005 has free gpu_card=$free"; exit 0; }
for j in $OLD; do
  st=$(qstat -u USER | awk -v j=$j "\$1==j{print \$5}" | sort -u | tr "\n" " ")
  case "$st" in *r*|*t*) echo "MOVE_ABORT job $j already running ($st)"; exit 1 ;; esac
  [ -n "$st" ] || { echo "MOVE_ABORT job $j not in queue"; exit 1; }
done
for tag in apple lemon; do
  [ -z "$(find $O/e13/pi0_l2_$tag -name result.json 2>/dev/null | head -1)" ] || { echo "MOVE_ABORT cells exist for $tag"; exit 1; }
done
qdel $OLD
cd ${SE_VLA_ROOT}
NEW=()
for spec in "1 apple 0 33200" "2 apple 0 36300" "1 lemon 1 45100" "2 lemon 1 48200"; do
  read -r s TAG TID PB <<< "$spec"
  out=$(qsub -terse -N RvE13 -q gpu@HOST_A -t 1-150 -tc 2 -o $J -hold_jid $FOLD \
    -v ARM=pi0_q05_s$s,CK=$O/e13/ckpt/pi0_q05_s$s/offset_0,SUITE=safety_static_obstacles,LEVEL=2,TID=$TID,NOFF=50,REPS=3,OUTROOT=$O/e13/pi0_l2_$TAG,EXPECT_HOST=HOST_A,PORTBASE=$PB,SHIELD=observe,EXPORT_SUITE=1,PI0=1 $S/eval.sh)
  jid=${out%%.*}; NEW+=($jid); echo "E13_l2${TAG}_s${s}_moved004 $jid" | tee -a $IDS
done
KEEP=$(grep -E "^E13_(l1t|l2mango|l2onion|l2tomato)" $IDS | awk "{print \$2}" | tr "\n" ",")
qalter -hold_jid "${KEEP}$(IFS=,; echo "${NEW[*]}")" $CLEAN
echo "MOVE_DONE new=${NEW[*]}"
