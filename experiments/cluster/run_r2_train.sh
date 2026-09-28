#!/usr/bin/env bash
#$ -N R2_train
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 4
#$ -t 1-2
#$ -tc 2
#$ -o ${WORK_ROOT}/lora_dagger/se_r2/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/se_r2/batch/j.$JOB_ID.$TASK_ID.out
#
# THE ACTUAL CURRICULUM. Everything before this in the SEC line trained a fixed subset ONCE
# and compared subsets -- that is easy-only training, not a curriculum. A curriculum stages
# the data ACROSS training: learn the easy part, then CONTINUE the same policy on a harder
# release. `main_phase2` re-inits from config.weight_loader on every call and never resumes,
# so continuation is done by folding phase 1's LoRA back into a full checkpoint and using it
# as phase 2's base (SE_VLA_TRAIN_BASE_CHECKPOINT, added today; default unchanged).
#
#   CURRICULUM   phase1 = s1   (225 deltas, S1_early)        -> fold -> phase2 = s1s2
#   STATIC       phase1 = s1s2 (290 deltas, S1_early+S2_mid)  -> fold -> phase2 = s1s2
#
# Both arms: two phases, one fold each, 100 + 100 steps, identical phase-2 data, identical
# starting checkpoint, identical optimizer-reset count. The ONLY difference is whether
# phase 1 withheld S2_mid. That isolates release ORDER, which is the thing "easy to hard"
# actually claims -- and which no earlier arm in this project ever varied.
#
# Records: 3657 in every set, deltas differing only by stage. Failure-trajectory records
# whose target would be their own (crash-causing) action are dropped, not zeroed -- the R1
# defect that cost 6-10 successes per arm.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OP=external/VLA-Arena/vla_arena/models/openpi
SEC=$LR/se_r2
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python
STEPS=${STEPS:-100}
ORIG_BASE=${WORK_ROOT}/lora_dagger/se_curr/ckpt/CURRICULUM_p2/offset_0

case "$SGE_TASK_ID" in
  1) ARM=CURRICULUM; P1=s1   ;;
  2) ARM=STATIC;     P1=s1s2 ;;
  *) echo "no arm"; exit 0 ;;
esac
P2=s1s2
mkdir -p "$SEC/batch"

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
  echo "--- $ARM $4: records=$1 base=$2 steps=$STEPS"
  SE_VLA_TRAIN_BASE_CHECKPOINT="$2" PYTHONPATH=src:external/VLA-Arena:. $VENV \
    $OP/scripts/train_phase2_lora.py \
    --offset 0 --records-root "$1/blobs" --derived-root "$1/derived/folds" \
    --batch-size 2 --quiet-weight 0.0 \
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
  train_phase "$SEC/${P1}_records" "$ORIG_BASE" "$A1" "phase1($P1)"
fi
if [ ! -d "$C1/params" ]; then fold_it "$A1" "$F1"; fi
[ -d "$C1/params" ] || { echo "CURR_ABORT: phase1 fold produced no params at $C1"; exit 1; }
[ -e "$C1/assets" ] || ln -s "$ORIG_BASE/assets" "$C1/assets"

# ---------------- phase 2 ----------------
A2=$SEC/adapter/${ARM}_p2
F2=$SEC/ckpt/${ARM}_p2
C2=$F2/offset_0
if [ ! -s "$A2/offset_0/metrics.json" ]; then
  train_phase "$SEC/${P2}_records" "$C1" "$A2" "phase2($P2)"
fi
if [ ! -d "$C2/params" ]; then fold_it "$A2" "$F2"; fi
[ -d "$C2/params" ] || { echo "CURR_ABORT: phase2 fold produced no params at $C2"; exit 1; }
[ -e "$C2/assets" ] || ln -s "$ORIG_BASE/assets" "$C2/assets"

echo "CURRICULUM_ARM_DONE arm=$ARM p1=$P1 p2=$P2 final=$C2"
