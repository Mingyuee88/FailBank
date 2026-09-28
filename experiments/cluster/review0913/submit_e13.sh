#!/usr/bin/env bash
# Review 2026-09-16, E13: two more pi0 FailBank training orders (data seed 1, 2) so the pi0 rows
# report the same three-order mean as pi0.5. Recipe copied from PI0_Q05 (data seed 0):
# se_pi0_q05/s1s2_records, pi0_col_t2 blobs, lambda_q 0.5, 800 steps, batch 32, first-action loss,
# pi0 base, asset VLA-Arena/VLA_Arena_L0_L_lerobot_openpi, guard 1.10 / 0.05 (unchanged).
set -euo pipefail
S=${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913
LR=${WORK_ROOT}/lora_dagger
O=$LR/review0913
J=${SE_VLA_ROOT}/joblogs
IDS=$O/lists/job_ids.txt
TRAINQ="gpu@HOST_A,gpu@HOST_B,gpu@HOST_C,gpu@HOST_D,gpu@HOST_E"
ROOT=$LR/se_pi0_q05/s1s2_records
BLOBS=$LR/pi0_col_t2/records/blobs
BASE=${SE_VLA_ROOT}/checkpoints/pi0_vla_arena_finetuned
CFG=pi0_vla_arena_low_mem_finetune
ASSET=VLA-Arena/VLA_Arena_L0_L_lerobot_openpi
[ -e $O/e13 ] && { echo "E13_ABORT e13 exists"; exit 1; }
python3 - "$LR/se_pi0_q05/adapter/PI0_Q05/offset_0/metrics.json" "$ROOT" "$BLOBS" "$BASE" "$ASSET" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); root, blobs, base, asset = sys.argv[2:]
chk = {"train_manifest": root + "/derived/folds/offset_0/train.jsonl", "records_root": blobs,
       "quiet_weight": 0.5, "best_step": 800, "base_checkpoint": base, "asset_id": asset, "data_seed": 0}
bad = {k: (m.get(k), v) for k, v in chk.items() if m.get(k) != v}
if bad: print("E13_ABORT recipe mismatch", bad); sys.exit(1)
print("E13_RECIPE_OK")
PY
cd ${SE_VLA_ROOT}
sub() { local name=$1; shift; local out; out=$(qsub -terse "$@"); local jid=${out%%.*}; echo "$name $jid" | tee -a "$IDS" >&2; echo "$jid"; }
mkdir -p $O/e13
: > $O/lists/e13_train.txt; : > $O/lists/e13_fold.txt; : > $O/lists/cleanup_e13.txt
for s in 1 2; do
  echo "pi0_q05_s$s $ROOT $BLOBS $O/e13/adapter/pi0_q05_s$s 0.5 $s 800 $CFG $BASE $ASSET 1" >> $O/lists/e13_train.txt
  echo "$O/e13/adapter/pi0_q05_s$s $O/e13/ckpt/pi0_q05_s$s $BASE" >> $O/lists/e13_fold.txt
  echo "$O/e13/ckpt/pi0_q05_s$s" >> $O/lists/cleanup_e13.txt
done
T=$(sub E13_train -N RvE13Tr -q "$TRAINQ" -t 1-2 -tc 2 -o $J -v LIST=$O/lists/e13_train.txt $S/train.sh)
F=$(sub E13_fold -N RvE13Fold -q "$TRAINQ" -t 1-2 -tc 2 -o $J -hold_jid $T -v LIST=$O/lists/e13_fold.txt $S/fold.sh)
EV=()
ev() {  # name host level tid reps outroot arm ck portbase range
  EV+=($(sub "$1" -N RvE13 -q gpu@$2 -t "${10}" -tc 2 -o $J -hold_jid $F \
    -v ARM=$7,CK=$8,SUITE=safety_static_obstacles,LEVEL=$3,TID=$4,NOFF=50,REPS=$5,OUTROOT=$6,EXPECT_HOST=$2,PORTBASE=$9,SHIELD=observe,EXPORT_SUITE=1,PI0=1 \
    $S/eval.sh))
}
for s in 1 2; do
  CK=$O/e13/ckpt/pi0_q05_s$s/offset_0
  for t in 0 1 2 3 4; do
    PB=$(( s == 1 ? 20000 + 1000 * t : 40000 + 1000 * t ))
    ev E13_l1t${t}_s$s HOST_A 1 $t 1 $O/e13/pi0_l1_t$t pi0_q05_s$s $CK $PB 1-50
  done
  OFFP=$(( (s - 1) * 3100 ))
  for spec in "0 apple HOST_B 20000" "1 lemon HOST_B 29300" "2 mango HOST_D 20000" "3 onion HOST_D 29300" "4 tomato HOST_A 26000"; do
    read -r TID TAG HOST PB <<< "$spec"
    ev E13_l2${TAG}_s$s $HOST 2 $TID 3 $O/e13/pi0_l2_$TAG pi0_q05_s$s $CK $((PB + OFFP)) 1-150
  done
done
HOLD=$(IFS=,; echo "${EV[*]}")
sub CLEAN_E13 -N RvClean -q long -o $J -hold_jid "$HOLD" -v LIST=$O/lists/cleanup_e13.txt $S/cleanup_ckpt.sh >/dev/null
echo "E13_SUBMITTED"
