#!/usr/bin/env bash
#$ -N NoCurr
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
# (host constraint removed: training needs no pinned host; only evaluation does)
#$ -pe smp 8
#$ -t 1-4
#$ -tc 2
#$ -o ${WORK_ROOT}/lora_dagger/nocurr/batch/
#$ -j y
#
# NO CURRICULUM + LEARNING FROM SUCCESSES. Two changes to the recipe, both motivated.
#
# 1) Drop the artificial staging. Measured: phase1 trains s1, phase2 trains s1s2, and those
#    two sets are 97%+ byte-identical (R1 differs on 65/3657 rows, R2 on 159/6535, R3 on
#    88/3364). CURRICULUM vs STATIC tied at 424 cells/arm (p=0.845). So the "stages" were a
#    no-op; this trains once on the full set for the same total budget (800 steps).
#
# 2) Give the quiet records a weight. In derived_lora_data.py the sample weight is
#        triggered -> teacher weight ;  quiet -> quiet_weight ;  else -> 0
#    and every round so far passed --quiet-weight 0.0, so 56-58% of the data contributed
#    NOTHING (R1 2129/3657, R2 3672/6535, R3 1905/3364 rows are quiet). Those are the
#    successful, non-intervened steps -- exactly the "learn from the successes" signal, and
#    simultaneously the replay anchor that keeps a new round from overwriting the last one.
#
#    This matters most for round 3: it has only 1459 triggered records, which is why 400
#    steps overfit past the 0.05 quiet-drift limit and the adapter was refused. Weighting
#    the quiet half raises the effective data 2.3x.
#
# Four arms: {R2, R3} x {quiet 0.0, quiet 0.2}. The quiet=0.0 arms are the controls that
# isolate the staging removal on its own.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OP=external/VLA-Arena/vla_arena/models/openpi
SEC=${OUT_ROOT:-$LR/nocurr}
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python
STEPS=${STEPS:-800}
BATCH=${BATCH:-32}
ORIG_BASE=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned

case "$SGE_TASK_ID" in
  1) ARM=R2_q00; SRC=$LR/se_r2; QW=0.0 ;;
  2) ARM=R2_q20; SRC=$LR/se_r2; QW=0.2 ;;
  3) ARM=R3_q00; SRC=$LR/se_r3; QW=0.0 ;;
  4) ARM=R3_q20; SRC=$LR/se_r3; QW=0.2 ;;
  *) echo "no arm"; exit 0 ;;
esac
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
    ${DATA_SEED:+--data-seed $DATA_SEED} \
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
