#!/bin/bash
#$ -N RndCurve
#$ -cwd
#$ -q gpu@HOST_A
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-3
#$ -tc 1
#$ -j y
# Round curve at FIXED quiet_weight=0.2.
#
# WHY q=0.2 and not 0.0: run_nocurr_train.sh defines {R2,R3} x {q00,q20}, but only
# R2_q00, R2_q20 and R3_q20 exist -- R3_q00 never produced an adapter because round 3
# fails the acceptance guard at quiet_weight 0. Holding the weight fixed is the only way
# the curve measures rounds rather than rounds-and-weight together.
#
# Banks are the ACCUMULATED ones (bank_r3d = B_2 u D(pi_2), bank_r4d = B_3 u D(pi_3)),
# not the non-accumulating se_r3 that the old round-3 run used.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
cd ${SE_VLA_ROOT}
L=${WORK_ROOT}/lora_dagger
OP=external/VLA-Arena/vla_arena/models/openpi
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python
ORIG_BASE=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
STEPS=${STEPS:-800}; BATCH=${BATCH:-32}; QW=0.2
SEC=$L/round_curve

case "$((SGE_TASK_ID-1))" in
  # R1's bank is se_curr/s1s2_records: the curriculum scripts differ in training ORDER
  # (phase1 = s1 vs s1s2), not in data -- all their sets hold the same 3657 rows, which are
  # exactly the bank_round=r1 rows inside se_r2. Trained single-phase here, so no curriculum.
  0) ARM=R1_q20;  SRC=$L/se_curr/s1s2_records ;;
  1) ARM=R3c_q20; SRC=$L/bank_r3d ;;
  2) ARM=R4_q20;  SRC=$L/bank_r4d ;;
  *) echo "no arm"; exit 0 ;;
esac
f=$SRC/derived/folds/offset_0/train.jsonl
[ -s "$f" ] || { echo "RC_ABORT: no records at $f"; exit 1; }
mkdir -p "$SEC"
echo "    arm=$ARM src=$SRC quiet_weight=$QW rows=$(wc -l < "$f") steps=$STEPS"

A1=$SEC/adapter/${ARM}; F1=$SEC/ckpt/${ARM}; C1=$F1/offset_0
if [ ! -s "$A1/offset_0/metrics.json" ]; then
  SE_VLA_TRAIN_BASE_CHECKPOINT="$ORIG_BASE" PYTHONPATH=src:external/VLA-Arena:. $VENV \
    $OP/scripts/train_phase2_lora.py \
    --offset 0 --records-root "$SRC/blobs" --derived-root "$SRC/derived/folds" \
    --batch-size "$BATCH" --quiet-weight "$QW" \
    --smoke-steps "$STEPS" --validation-interval "$STEPS" --patience 99 \
    --output-root "$A1"
fi
$VENV - "$A1" "$ARM" <<'PYV'
import json, sys, pathlib
out, arm = sys.argv[1], sys.argv[2]
m = json.load(open(pathlib.Path(out) / "offset_0/metrics.json"))
print(f"GUARD {arm}: flow_ratio={m.get('best_quiet_flow_ratio')} "
      f"drift={m.get('best_quiet_action_drift')} accepted={m.get('accepted')}")
PYV
# A guard rejection is a RESULT here (over-optimisation), not a crash: record and stop.
if ! $VENV -c "
import json,sys,pathlib
m=json.load(open(pathlib.Path('$A1')/'offset_0/metrics.json'))
sys.exit(0 if m.get('accepted') else 1)"; then
  echo "RC_REJECTED arm=$ARM -- guard refused this round; this is the curve's ceiling"
  exit 0
fi
if [ ! -d "$C1/params" ]; then
  PYTHONPATH=src:external/VLA-Arena:. $VENV - "$A1" "$F1" "$ORIG_BASE" <<'PYFOLD'
import pathlib, sys
OP = pathlib.Path("${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/models/openpi")
sys.path.insert(0, str(OP / "scripts"))
import fold_lora_adapter as F
import orbax.checkpoint as ocp
F.FOLDS = pathlib.Path(sys.argv[1]); F.OUT = pathlib.Path(sys.argv[2])
F.BASE = pathlib.Path(sys.argv[3]).resolve()   # never rely on the module default
print("FOLD_BASE=%s" % F.BASE)
F.OUT.mkdir(parents=True, exist_ok=True)
F.fold_one(0, ocp.StandardCheckpointer(), verbose=True)
print("FOLD_OK")
PYFOLD
fi
[ -d "$C1/params" ] || { echo "RC_ABORT: fold produced no params at $C1"; exit 1; }
[ -e "$C1/assets" ] || ln -s "$ORIG_BASE/assets" "$C1/assets"
$VENV roundG/pi05_stage2/lora/verify_family.py "$C1/params" pi05
echo "RC_DONE arm=$ARM quiet=$QW final=$C1"
