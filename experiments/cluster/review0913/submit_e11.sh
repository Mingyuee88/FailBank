#!/usr/bin/env bash
# Review 2026-09-15, E11: complete the pi0 rows of the main table.
#  (a) pi0 + AEGIS on L1 T0-T4, 50 x 1 condition, HOST_A (pairs with pi0_eval2 base / FailBank)
#  (b) pi0 on L2 T0-T4: base / AEGIS / FailBank, 50 x 3 conditions (the L2 protocol)
# AEGIS arms wait for the GLM probe and then for two smoke cells to pass the gate.
set -euo pipefail
S=${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913
L=${SE_VLA_ROOT}/roundG/pi05_stage2/lora
LR=${WORK_ROOT}/lora_dagger
O=$LR/review0913
J=${SE_VLA_ROOT}/joblogs
IDS=$O/lists/job_ids.txt
ST=safety_static_obstacles
FB=$LR/se_pi0_q05/ckpt_pi0fold_1405748/PI0_Q05/offset_0
[ -d "$FB/params" ] || { echo "E11_ABORT no pi0 FailBank checkpoint"; exit 1; }
cd ${SE_VLA_ROOT}
sub() { local name=$1; shift; local out; out=$(qsub -terse "$@"); local jid=${out%%.*}; echo "$name $jid" | tee -a "$IDS" >&2; echo "$jid"; }
ev() {  # name host tid reps outroot arm ck shield portbase hold task-range
  local hold=()
  [ -n "${10}" ] && hold=(-hold_jid "${10}")
  sub "$1" -N RvE11 -q gpu@$2 -t "${11}" -tc 2 -o $J "${hold[@]}" \
    -v ARM=$6,CK=$7,SUITE=$ST,LEVEL=$8,TID=$3,NOFF=50,REPS=$4,OUTROOT=$5,EXPECT_HOST=$2,PORTBASE=$9,SHIELD=${12},EXPORT_SUITE=1,PI0=1 \
    $S/eval.sh
}
# GLM service + readiness probe
if qstat -u USER 2>/dev/null | grep -q glm45v; then GLM=$(qstat -u USER | awk '/glm45v/{print $1; exit}'); echo "GLM already queued $GLM" >&2
else GLM=$(cd $L && sub E11_glm_serve $L/run_glm_serve_a6k.sh); fi
P=$(sub E11_glm_probe -q long -o $J $S/glm_probe.sh)
# smoke cells (task 1 of the real arrays, so they count)
SM1=$(ev E11_smoke_l1t0_aegis HOST_A 0 1 $O/e11/pi0_l1_t0 aegis - 1 20000 $P 1-1 aegis)
SM2=$(ev E11_smoke_l2apple_aegis HOST_B 0 3 $O/e11/pi0_l2_apple aegis - 2 23100 $P 1-1 aegis)
G=$(sub E11_gate -q long -o $J -hold_jid $SM1,$SM2 -v CELLS=$O/e11/pi0_l1_t0/aegis/off0_r0,$O/e11/pi0_l2_apple/aegis/off0_r0 $S/cell_gate.sh)
# (a) pi0 + AEGIS, L1 T0-T4 on l40s-004
for t in 0 1 2 3 4; do
  ev E11_l1t${t}_aegis HOST_A $t 1 $O/e11/pi0_l1_t$t aegis - 1 $((20000 + 1000 * t)) $G 1-50 aegis >/dev/null
done
# (b) pi0 on L2 T0-T4
for spec in "0 apple HOST_B 20000" "1 lemon HOST_B 29300" "2 mango HOST_D 20000" "3 onion HOST_D 29300" "4 tomato HOST_A 26000"; do
  read -r TID TAG HOST PB <<< "$spec"
  R=$O/e11/pi0_l2_$TAG
  ev E11_l2${TAG}_base     $HOST $TID 3 $R base     -   2 $PB              "" 1-150 observe >/dev/null
  ev E11_l2${TAG}_failbank $HOST $TID 3 $R failbank $FB 2 $((PB + 6200))  "" 1-150 observe >/dev/null
  ev E11_l2${TAG}_aegis    $HOST $TID 3 $R aegis    -   2 $((PB + 3100))  $G 1-150 aegis >/dev/null
done
echo "E11_SUBMITTED"
