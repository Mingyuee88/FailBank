#!/usr/bin/env bash
#$ -N R1b_train
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 8
#$ -o ${WORK_ROOT}/lora_dagger/se_curr1b/batch/
#$ -j y
#
# RETRAIN ROUND 1 AT ROUND 2's TRAINING BUDGET.
#
# The r1 vs r2 comparison had an uncontrolled variable. From the saved metrics:
#     r1 (se_curr)   best_step=100   run_curriculum.sh  STEPS default 100
#     r2 (se_curr2)  best_step=400   run_curriculum2.sh STEPS default 400
# r2 trained four times as long, so the measured r1->r2 gain (dSR +0.088, p=0.016) cannot
# be attributed to self-evolution while that difference stands.
#
# This retrains round 1 on its OWN records (se_curr/s1_records, 3657 rows, collected by
# pi0) at steps=400 batch=32, identical to round 2. Afterwards r1b and r2 differ in exactly
# one respect: which policy generated the training data.
#
# Data volume differs by round -- R1 3657, R2 6535, R3 3364 rows -- but that is ENDOGENOUS:
# a stronger policy fails less and yields fewer records. It is part of what the rounds claim
# is about, unlike the step count, which was just an inconsistent default.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OP=external/VLA-Arena/vla_arena/models/openpi
SEC=$LR/se_curr1b
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python
STEPS=${STEPS:-400}
BATCH=${BATCH:-32}
ORIG_BASE=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned

ARM=CURRICULUM; P1=s1
P2=s1s2
mkdir -p "$SEC/batch"

R1=$LR/se_curr
for MODE in s1 s1s2; do
  f=$R1/${MODE}_records/derived/folds/offset_0/train.jsonl
  [ -s "$f" ] || { echo "R1B_ABORT: $MODE record set missing at $f"; exit 1; }
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
  train_phase "$R1/${P1}_records" "$ORIG_BASE" "$A1" "phase1($P1)"
fi
if [ ! -d "$C1/params" ]; then fold_it "$A1" "$F1"; fi
[ -d "$C1/params" ] || { echo "CURR_ABORT: phase1 fold produced no params at $C1"; exit 1; }
[ -e "$C1/assets" ] || ln -s "$ORIG_BASE/assets" "$C1/assets"

# ---------------- phase 2 ----------------
A2=$SEC/adapter/${ARM}_p2
F2=$SEC/ckpt/${ARM}_p2
C2=$F2/offset_0
if [ ! -s "$A2/offset_0/metrics.json" ]; then
  train_phase "$R1/${P2}_records" "$C1" "$A2" "phase2($P2)"
fi
if [ ! -d "$C2/params" ]; then fold_it "$A2" "$F2"; fi
[ -d "$C2/params" ] || { echo "CURR_ABORT: phase2 fold produced no params at $C2"; exit 1; }
[ -e "$C2/assets" ] || ln -s "$ORIG_BASE/assets" "$C2/assets"

echo "R1B_TRAIN_DONE arm=$ARM p1=$P1 p2=$P2 final=$C2"
