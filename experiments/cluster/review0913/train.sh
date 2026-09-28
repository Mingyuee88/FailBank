#!/usr/bin/env bash
#$ -N RvTrain
#$ -cwd
#$ -l gpu_card=1
#$ -pe smp 8
#$ -j y
# Review 2026-09-13. Guarded LoRA update; the trainer invocation is identical to
# run_nocurr_train.sh train_phase (batch 32, --smoke-steps N, one validation at N, patience 99).
# LIST line: ARM RECORDS_ROOT BLOBS OUT QW DATA_SEED STEPS LORA_CONFIG BASE_CKPT ASSET_ID SUPERVISE_K
#   ASSET_ID "-" keeps the trainer default (pi0.5 asset layout).
#   SUPERVISE_K 1 is the stock first-action loss (verified bit-identical after the E8 patch).
set -euo pipefail
: "${LIST:?set LIST}"
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 WANDB_MODE=disabled
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
cd ${SE_VLA_ROOT}
OP=external/VLA-Arena/vla_arena/models/openpi
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python
read -r ARM ROOT BLOBS OUTA QW DSEED STEPS LCFG BASE ASSETID SUPK <<< "$(sed -n "${SGE_TASK_ID}p" "$LIST")"
echo "=== RvTrain arm=$ARM root=$ROOT qw=$QW seed=$DSEED steps=$STEPS cfg=$LCFG k=$SUPK host=$(hostname) ==="
F=$ROOT/derived/folds/offset_0/train.jsonl
[ -s "$F" ] || { echo "TRAIN_ABORT no train manifest $F"; exit 1; }
[ -d "$BLOBS" ] || { echo "TRAIN_ABORT no blobs $BLOBS"; exit 1; }
if [ -s "$OUTA/offset_0/metrics.json" ] || [ -s "$OUTA/offset_0/metrics_rejected.json" ]; then
  echo "TRAIN_SKIP $OUTA already has a verdict"; exit 0
fi
echo "    rows=$(wc -l < "$F")"
extra=()
if [ "$ASSETID" != "-" ]; then extra=(--assets-dir "$BASE/assets" --asset-id "$ASSETID"); fi
set +e
SE_VLA_TRAIN_BASE_CHECKPOINT="$BASE" SE_VLA_TRAIN_LORA_CONFIG="$LCFG" SE_VLA_SUPERVISE_FIRST_K="$SUPK" \
PYTHONPATH=src:external/VLA-Arena:. $VENV $OP/scripts/train_phase2_lora.py \
  --offset 0 --records-root "$BLOBS" --derived-root "$ROOT/derived/folds" \
  --batch-size 32 --quiet-weight "$QW" --data-seed "$DSEED" \
  --smoke-steps "$STEPS" --validation-interval "$STEPS" --patience 99 \
  --output-root "$OUTA" ${extra[@]+"${extra[@]}"}
rc=$?
set -e
echo "TRAIN_EXIT=$rc arm=$ARM"
$VENV - "$OUTA" "$ARM" <<"PY"
import json, pathlib, sys
d = pathlib.Path(sys.argv[1]) / "offset_0"
for name in ("metrics.json", "metrics_rejected.json"):
    p = d / name
    if p.exists():
        m = json.load(open(p))
        print("TRAIN_VERDICT arm=%s file=%s accepted=%s ratio=%s drift=%s triggered=%s base_flow=%s manifest=%s" % (
            sys.argv[2], name, m.get("accepted"), m.get("best_quiet_flow_ratio"),
            m.get("best_quiet_action_drift"), m.get("best_triggered_loss"),
            m.get("base_quiet_flow_loss"), m.get("train_manifest")))
        break
else:
    print("TRAIN_VERDICT arm=%s file=NONE" % sys.argv[2])
    sys.exit(1)
PY
