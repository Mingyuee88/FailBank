#!/usr/bin/env bash
#$ -N DynTrain
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -pe smp 4
#$ -j y
#
# Train the dynamic-suite arm on safety_dynamic_obstacles level_2 task 0.
#
# Task choice is measured, not assumed: a 64-cell discriminability scan over the whole suite
# found level_1 unusable (t1/t2 at SR 100 with zero policy cost, t3 at SR 37.5 with zero
# policy cost, t4 thick signal but SR 25) and level_2 t0 the only task with both headroom
# and policy-induced cost. On the full 47-cell collection that headroom shrank -- SR 80.9,
# policy cost on 5/47 cells -- so the harvest here is thinner than any static task we trained
# on. What makes training viable anyway is the shield trace: 1454 of 7702 steps (18.9%) are
# triggered, spread over 47/47 episodes.
#
# Recipe is the one that survived: no curriculum staging, quiet_weight 0, single 800-step
# phase. Staging was disproven (424 cells/arm, p=0.845, only 2.4% of rows differ).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OP=external/VLA-Arena/vla_arena/models/openpi
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python
ORIG_BASE=${BASE_CK:-${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned}
# A silent fallback here trained a pi0 run as pi0.5 and went unnoticed until the
# adapter failed to load: the job log printed the pi0 config name (intent) while the
# weights came out with pi05 structure (state_proj/action_time_mlp absent, adaRMS
# Dense_0 present). Require the caller to be explicit instead of defaulting.
: "${LORA_CFG:?LORA_CFG must be set explicitly (e.g. pi05_vla_arena_low_mem_finetune \
or pi0_vla_arena_low_mem_finetune) -- no silent default}"
export SE_VLA_TRAIN_LORA_CONFIG=$LORA_CFG
case "$SE_VLA_TRAIN_LORA_CONFIG" in
  pi0_*)  EXPECT_FAMILY=pi0  ;;
  pi05_*) EXPECT_FAMILY=pi05 ;;
  *) echo "UNKNOWN_LORA_CONFIG_FAMILY $SE_VLA_TRAIN_LORA_CONFIG"; exit 1 ;;
esac
export SE_VLA_EXPECT_FAMILY=$EXPECT_FAMILY
ASSET_ARGS=""
[ -n "${ASSETS_DIR:-}" ] && ASSET_ARGS="$ASSET_ARGS --assets-dir $ASSETS_DIR"
[ -n "${ASSET_ID:-}" ] && ASSET_ARGS="$ASSET_ARGS --asset-id $ASSET_ID"

# RAW_ROOT/BASE_CK/LORA_CFG let the same pipeline train a different base model. Unset, this
# is the Pi0.5 dynamic arm exactly as before. For the Pi0 arm all three change together:
# its records, its checkpoint, and the LoRA config Arena ships for it (which differs from the
# Pi0.5 one in extra_delta_transform=True and action_horizon=50 -- mixing them would
# mis-scale every action).
RAW=${RAW_ROOT:-$LR/dcol_L2t0/records}
# DYN_ROOT/ARM_NAME let a second arm train off a pre-split record set (the holdout, offsets
# 0-31 only) without rebuilding anything. Unset, this is the full-offset arm as before.
# The full-offset arm is in-sample by construction -- collection walked 0..47 and evaluation
# walks 0..49 -- so it is an upper bound, not a generalisation number. The holdout arm is
# the one that answers "does it transfer to unseen initial states".
DYN=${DYN_ROOT:-$LR/se_dyn_L2t0}
ARM=${ARM_NAME:-DYN_L2t0}
QW=${QUIET_W:-0.0}
STEPS=${STEPS:-800}
BATCH=${BATCH:-32}
mkdir -p "$DYN"

echo "########## 1. build record set ##########"
if [ ! -s "$DYN/s1s2_records/derived/folds/offset_0/train.jsonl" ]; then
  echo "--- annotating raw records (stage labels + steps_to_first_risk)"
  PYTHONPATH=src:external/VLA-Arena:. $VENV roundG/pi05_stage2/lora/build_derived_c1.py \
    --records-root "$RAW"
  echo "--- building s1s2 record set"
  PYTHONPATH=src:external/VLA-Arena:. $VENV roundG/pi05_stage2/lora/build_round_records.py \
    --src-root "$RAW" --dst-root "$DYN/s1s2_records" --mode s1s2
fi
F=$DYN/s1s2_records/derived/folds/offset_0/train.jsonl
[ -s "$F" ] || { echo "DYN_ABORT: no record set at $F"; exit 1; }
ROWS=$(wc -l < "$F")
echo "    arm=$ARM rows=$ROWS quiet_weight=$QW steps=$STEPS"
echo "    base=$ORIG_BASE"
echo "    lora_config=$SE_VLA_TRAIN_LORA_CONFIG"
echo "    quiet_weight=$QW"
echo "    records=$RAW"
echo "    asset_args=${ASSET_ARGS:-<trainer defaults>}"
[ "$ROWS" -ge 200 ] || { echo "DYN_ABORT: only $ROWS rows, too thin to train"; exit 1; }

fold_it () {  # $1=adapter dir (contains offset_0/params)  $2=output ckpt dir
  SE_VLA_FOLD_BASE="${ORIG_BASE}" \
  PYTHONPATH=src:external/VLA-Arena:. $VENV - "$1" "$2" <<'PYFOLD'
import pathlib, sys
OP = pathlib.Path("${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/models/openpi")
sys.path.insert(0, str(OP / "scripts"))
import fold_lora_adapter as F
import orbax.checkpoint as ocp
# F.BASE defaults to the pi05 checkpoint inside fold_lora_adapter.py. Overriding only
# FOLDS/OUT (as this did) folds a pi0 LoRA delta into the pi05 base and silently
# produces a pi05-structured checkpoint: the adapter is fine, the merge is wrong.
import os
F.FOLDS = pathlib.Path(sys.argv[1]); F.OUT = pathlib.Path(sys.argv[2])
if os.environ.get("SE_VLA_FOLD_BASE"):
    F.BASE = pathlib.Path(os.environ["SE_VLA_FOLD_BASE"]).resolve()
print("FOLD_BASE=%s" % F.BASE)
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
    --output-root "$3" $ASSET_ARGS
  $VENV - "$3" "$STEPS" "$ARM" "$4" "$2" <<'PYV'
import json, sys, pathlib
out, steps, arm, label, base = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
m = json.load(open(pathlib.Path(out) / "offset_0/metrics.json"))
print(f"{arm} {label}: best_step={m.get('best_step')} validations_run={m.get('validations_run')} "
      f"flow_ratio={m.get('best_quiet_flow_ratio'):.5f} act_drift={m.get('best_quiet_action_drift'):.6f} "
      f"accepted={m.get('accepted')} base={m.get('base_checkpoint')}")
assert m.get("best_step") == steps and m.get("validations_run") == 1
assert str(m.get("base_checkpoint")) == str(pathlib.Path(base).resolve()), \
    f"base mismatch: metrics say {m.get('base_checkpoint')}, expected {base}"
print("PHASE_VERIFIED")
PYV
}

echo
echo "########## 2. train ##########"
A1=$DYN/adapter/${ARM}
F1=$DYN/ckpt/${ARM}
C1=$F1/offset_0
if [ ! -s "$A1/offset_0/metrics.json" ]; then
  train_phase "$DYN/s1s2_records" "$ORIG_BASE" "$A1" "single($ARM)"
fi
if [ ! -d "$C1/params" ]; then fold_it "$A1" "$F1"; fi
[ -d "$C1/params" ] || { echo "DYN_ABORT: fold produced no params at $C1"; exit 1; }
[ -e "$C1/assets" ] || ln -s "$ORIG_BASE/assets" "$C1/assets"
echo "DYN_TRAIN_DONE arm=$ARM quiet=$QW rows=$ROWS final=$C1"
