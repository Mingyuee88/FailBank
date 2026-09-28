#!/usr/bin/env bash
#$ -N R3_train
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 8
#$ -o ${WORK_ROOT}/lora_dagger/se_curr3/batch/
#$ -j y
#
# ROUND 3 of self-evolution. Same recipe as rounds 1 and 2 -- only the record source moves.
#
#   base  SR 35.0 +-3.29  crash 14.6  polCC 3924
#   r1    SR 38.2 +-2.32  crash 11.6  polCC 3002   base->r1  dSR +0.064  p=0.043
#   r2    SR 42.6 +-1.62  crash  7.4  polCC 1987   r1->r2    dSR +0.088  p=0.016
#
# The gain grew rather than decayed, because each round trains on the failures the previous
# policy itself produced. A third point says whether that is a trend or a coincidence.
#
# One early warning is already visible in the collection run: r2 scored SR 89.4% while being
# recorded, with only 31 of 47 cells carrying any cost at all. Round 3 has materially less
# to learn from than round 2 did, so a smaller gain here would be expected, not surprising.
#
# Records come from r3_collect (r2 drove, shield observe-only, all 47 cells verified).
# Training starts from ORIG_BASE with the identical two-phase schedule, so r3 differs from
# r2 in exactly one respect: which policy generated its training data.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OP=external/VLA-Arena/vla_arena/models/openpi
SEC=$LR/se_curr3
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python
STEPS=${STEPS:-400}
BATCH=${BATCH:-32}
ORIG_BASE=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned

ARM=CURRICULUM; P1=s1
P2=s1s2
mkdir -p "$SEC/batch"

# ---------------- build the round-3 record sets ----------------
RAW=$LR/r3_collect/records
R3=$LR/se_r3
if [ ! -s "$R3/s1s2_records/derived/folds/offset_0/train.jsonl" ]; then
  echo "--- annotating raw r3 records (stage labels + steps_to_first_risk)"
  PYTHONPATH=src:external/VLA-Arena:. $VENV roundG/pi05_stage2/lora/build_derived_c1.py \
    --records-root "$RAW"
  for MODE in s1 s1s2; do
    echo "--- building $MODE record set"
    PYTHONPATH=src:external/VLA-Arena:. $VENV roundG/pi05_stage2/lora/build_round_records.py \
      --src-root "$RAW" --dst-root "$R3/${MODE}_records" --mode "$MODE"
  done
fi
for MODE in s1 s1s2; do
  f=$R3/${MODE}_records/derived/folds/offset_0/train.jsonl
  [ -s "$f" ] || { echo "R3_ABORT: $MODE record set missing at $f"; exit 1; }
  echo "    $MODE rows=$(wc -l < "$f")"
done

fold_it () {  # $1=adapter dir (contains offset_0/params)  $2=output ckpt dir
  PYTHONPATH=src:external/VLA-Arena:. $VENV - "$1" "$2" <<'PYFOLD'
import pathlib, sys
OP = pathlib.Path("${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/models/openpi")
sys.path.insert(0, str(OP / "scripts"))
import fold_lora_adapter as F
import orbax.checkpoint as ocp
F.FOLDS = pathlib.Path(sys.argv[1]); F.OUT = pathlib.Path(sys.argv[2])
F.OUT.mkdir(parents=True, exist_ok=True)
F.fold_one(0, ocp.StandardCheckpointer(), verbose=True)
print("FOLD_OK")
PYFOLD
}

train_phase () {  # $1=records root  $2=base ckpt  $3=output adapter dir  $4=label
  echo "--- $ARM $4: records=$1 base=$2 steps=$STEPS batch=$BATCH"
  SE_VLA_TRAIN_BASE_CHECKPOINT="$2" PYTHONPATH=src:external/VLA-Arena:. $VENV \
    $OP/scripts/train_phase2_lora.py \
    --offset 0 --records-root "$1/blobs" --derived-root "$1/derived/folds" \
    --batch-size "$BATCH" --quiet-weight 0.0 \
    --smoke-steps "$STEPS" --validation-interval "$STEPS" --patience 99 \
    --output-root "$3"
  $VENV - "$3" "$STEPS" "$ARM" "$4" "$2" <<'PYV'
import json, sys, pathlib
out, steps, arm, label, base = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
m = json.load(open(pathlib.Path(out) / "offset_0/metrics.json"))
print(f"{arm} {label}: best_step={m.get('best_step')} validations_run={m.get('validations_run')} "
      f"flow_ratio={m.get('best_quiet_flow_ratio'):.5f} act_drift={m.get('best_quiet_action_drift'):.6f} "
      f"accepted={m.get('accepted')} base={m.get('base_checkpoint')}")
assert m.get("best_step") == steps and m.get("validations_run") == 1
# the load-bearing check for a curriculum: phase 2 must have STARTED FROM PHASE 1,
# otherwise both phases trained from base and this is two independent runs, not a curriculum
assert str(m.get("base_checkpoint")) == str(pathlib.Path(base).resolve()), \
    f"phase base mismatch: metrics say {m.get('base_checkpoint')}, expected {base}"
print("PHASE_VERIFIED")
PYV
}

# fold_one(0, ...) writes into <OUT>/offset_0/{params,assets} and creates the assets link
# itself, so the checkpoint to hand onward is the offset_0 subdirectory, not its parent.
# (Passing the parent is what made the first attempt die with "File not found at .../params".)

# ---------------- phase 1 ----------------
A1=$SEC/adapter/${ARM}_p1
F1=$SEC/ckpt/${ARM}_p1
C1=$F1/offset_0
if [ ! -s "$A1/offset_0/metrics.json" ]; then
  train_phase "$R3/${P1}_records" "$ORIG_BASE" "$A1" "phase1($P1)"
fi
if [ ! -d "$C1/params" ]; then fold_it "$A1" "$F1"; fi
[ -d "$C1/params" ] || { echo "CURR_ABORT: phase1 fold produced no params at $C1"; exit 1; }
[ -e "$C1/assets" ] || ln -s "$ORIG_BASE/assets" "$C1/assets"

# ---------------- phase 2 ----------------
A2=$SEC/adapter/${ARM}_p2
F2=$SEC/ckpt/${ARM}_p2
C2=$F2/offset_0
if [ ! -s "$A2/offset_0/metrics.json" ]; then
  train_phase "$R3/${P2}_records" "$C1" "$A2" "phase2($P2)"
fi
if [ ! -d "$C2/params" ]; then fold_it "$A2" "$F2"; fi
[ -d "$C2/params" ] || { echo "CURR_ABORT: phase2 fold produced no params at $C2"; exit 1; }
[ -e "$C2/assets" ] || ln -s "$ORIG_BASE/assets" "$C2/assets"

echo "R3_TRAIN_DONE arm=$ARM p1=$P1 p2=$P2 final=$C2"
