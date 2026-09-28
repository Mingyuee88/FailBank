#!/usr/bin/env bash
#$ -N Tr_multi
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 8
#$ -j y
#$ -o ${WORK_ROOT}/lora_dagger/nocurr/batch/
#
# MULTI-TASK: the best recipe (no staging, quiet 0, 800 steps) trained on all four
# tasks at once instead of L1t2 alone. It reaches SR 90.8 / CC 18.5 on L1t2, but that
# number has never been shown to be a property of the recipe rather than of that one task.
#
# Note the composition before reading the result: L1t2 contributes 1039 of the 1305 deltas
# (80%), because it was collected with the weaker r1 policy while the other three were
# collected with R2_q00 (SR 85-89, so few failures). If the outcome improves only on L1t2,
# the fix is to re-collect the other tasks with a weaker policy, not to reweight.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OP=external/VLA-Arena/vla_arena/models/openpi
SEC=$LR/nocurr
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python
STEPS=${STEPS:-800}
BATCH=${BATCH:-32}
ORIG_BASE=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned

ARM=MULTI; SRC=$LR/se_multi; QW=0.0
P2=s1s2
mkdir -p "$SEC/batch"

f=$SRC/s1s2_records/derived/folds/offset_0/train.jsonl
[ -s "$f" ] || { echo "NC_ABORT: no record set at $f"; exit 1; }
echo "    arm=$ARM src=$SRC quiet_weight=$QW rows=$(wc -l < "$f") steps=$STEPS"

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
    --batch-size "$BATCH" --quiet-weight "$QW" \
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
    f"base mismatch: metrics say {m.get('base_checkpoint')}, expected {base}"
print("PHASE_VERIFIED")
PYV
}

# fold_one(0, ...) writes into <OUT>/offset_0/{params,assets} and creates the assets link
# itself, so the checkpoint to hand onward is the offset_0 subdirectory, not its parent.
# (Passing the parent is what made the first attempt die with "File not found at .../params".)

# ---------------- single phase, full set ----------------
A1=$SEC/adapter/${ARM}
F1=$SEC/ckpt/${ARM}
C1=$F1/offset_0
if [ ! -s "$A1/offset_0/metrics.json" ]; then
  train_phase "$SRC/s1s2_records" "$ORIG_BASE" "$A1" "single($ARM)"
fi
if [ ! -d "$C1/params" ]; then fold_it "$A1" "$F1"; fi
[ -d "$C1/params" ] || { echo "NC_ABORT: fold produced no params at $C1"; exit 1; }
[ -e "$C1/assets" ] || ln -s "$ORIG_BASE/assets" "$C1/assets"
echo "NOCURR_DONE arm=$ARM quiet=$QW final=$C1"
