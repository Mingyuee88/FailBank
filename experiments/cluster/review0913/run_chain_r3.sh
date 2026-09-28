#!/usr/bin/env bash
#$ -N ChainR3
#$ -cwd
#$ -l gpu_card=1
#$ -pe smp 8
#$ -j y
# E16: round 3 trained by CONTINUING from the round-2 adapter instead of restarting from
# the original base. Everything else is matched to the existing from-base arm
# nocurr/adapter/R3_q20: same bank (se_r3), 800 steps, batch 32, quiet_weight 0.2, data seed 0.
# START_CKPT is the folded round-2 checkpoint to continue from.
set -euo pipefail
: "${START_CKPT:?set START_CKPT to a folded checkpoint directory}"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OP=external/VLA-Arena/vla_arena/models/openpi
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python
ORIG_BASE=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
SRC=$LR/se_r3
OUT=$LR/chain_r3
ARM=R3_q20_chain
STEPS=800; BATCH=32; QW=0.2

[ -d "$START_CKPT/params" ] || { echo "CHAIN_ABORT no params at $START_CKPT"; exit 1; }
f=$SRC/s1s2_records/derived/folds/offset_0/train.jsonl
[ -s "$f" ] || { echo "CHAIN_ABORT no record set at $f"; exit 1; }
mkdir -p "$OUT/batch"
echo "    arm=$ARM start=$START_CKPT src=$SRC quiet_weight=$QW rows=$(wc -l < "$f") steps=$STEPS"

A1=$OUT/adapter/$ARM
if [ ! -s "$A1/offset_0/metrics.json" ]; then
  SE_VLA_TRAIN_BASE_CHECKPOINT="$START_CKPT" PYTHONPATH=src:external/VLA-Arena:. $VENV \
    $OP/scripts/train_phase2_lora.py \
    --offset 0 --records-root "$SRC/s1s2_records/blobs" \
    --derived-root "$SRC/s1s2_records/derived/folds" \
    --batch-size "$BATCH" --quiet-weight "$QW" \
    --smoke-steps "$STEPS" --validation-interval "$STEPS" --patience 99 \
    --output-root "$A1"
fi
$VENV - "$A1" "$STEPS" "$START_CKPT" <<'PYV'
import json, sys, pathlib
out, steps, start = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = json.load(open(pathlib.Path(out) / "offset_0/metrics.json"))
print("CHAIN_R3 best_step=%s validations_run=%s flow_ratio=%.5f act_drift=%.6f accepted=%s base=%s" % (
    m.get("best_step"), m.get("validations_run"), m.get("best_quiet_flow_ratio") or float("nan"),
    m.get("best_quiet_action_drift") or float("nan"), m.get("accepted"), m.get("base_checkpoint")))
assert str(m.get("base_checkpoint")) == str(pathlib.Path(start).resolve()), \
    "chain start mismatch: metrics say %s, expected %s" % (m.get("base_checkpoint"), start)
print("CHAIN_START_VERIFIED")
PYV
echo "CHAIN_R3_DONE adapter=$A1"
