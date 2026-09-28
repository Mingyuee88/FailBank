#!/usr/bin/env bash
#$ -N RvTrNG
#$ -cwd
#$ -l gpu_card=1
#$ -pe smp 8
#$ -j y
# Review 2026-09-14. DIAGNOSTIC ONLY, not a deployable adapter.
# E5 il_r1 and E6 r2only were rejected by the quiet guard at their single validation point
# (step 800 = last step; flow ratio 1.137 / 1.111 > 1.10). To evaluate their last-step
# weights, this reruns exactly the train.sh invocation with both guard limits raised to 1e9,
# so the step-800 validation is accepted and saved. Everything else is identical.
# The output directory gets a DIAGNOSTIC_NO_GUARD marker; metrics.json "accepted" is
# meaningless for these runs.
# LIST line: ARM RECORDS_ROOT BLOBS OUT QW DATA_SEED STEPS LORA_CONFIG BASE_CKPT ASSET_ID SUPERVISE_K
set -euo pipefail
: "${LIST:?set LIST}"
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 WANDB_MODE=disabled
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
cd ${SE_VLA_ROOT}
OP=external/VLA-Arena/vla_arena/models/openpi
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python
read -r ARM ROOT BLOBS OUTA QW DSEED STEPS LCFG BASE ASSETID SUPK <<< "$(sed -n "${SGE_TASK_ID}p" "$LIST")"
echo "=== RvTrainNoGuard arm=$ARM root=$ROOT qw=$QW seed=$DSEED steps=$STEPS cfg=$LCFG k=$SUPK host=$(hostname) ==="
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
  --quiet-flow-loss-ratio-limit 1e9 --quiet-action-drift-limit 1e9 \
  --output-root "$OUTA" ${extra[@]+"${extra[@]}"}
rc=$?
set -e
echo "TRAIN_EXIT=$rc arm=$ARM"
if [ -s "$OUTA/offset_0/metrics.json" ]; then
  echo "guard limits raised to 1e9 for a last-step diagnostic; not a deployable adapter" > "$OUTA/offset_0/DIAGNOSTIC_NO_GUARD"
fi
$VENV - "$OUTA" "$ARM" <<"PY"
import json, pathlib, sys
d = pathlib.Path(sys.argv[1]) / "offset_0"
p = d / "metrics.json"
if not p.exists():
    print("TRAIN_VERDICT arm=%s file=NONE" % sys.argv[2])
    sys.exit(1)
m = json.load(open(p))
print("TRAIN_VERDICT_DIAG arm=%s step=%s flow_ratio=%s drift=%s triggered=%s base_flow=%s (guard limits 1e9; would be rejected if flow>1.10 or drift>0.05)" % (
    sys.argv[2], m.get("best_step"), m.get("best_quiet_flow_ratio"),
    m.get("best_quiet_action_drift"), m.get("best_triggered_loss"), m.get("base_quiet_flow_loss")))
PY
