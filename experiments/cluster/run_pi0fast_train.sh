#!/usr/bin/env bash
#$ -N Pi0FTrain
#$ -cwd
#$ -l gpu_card=1
#$ -pe smp 8
#$ -j y
#
# Pi0-FAST arm: guarded LoRA update on the L1-t2 round-1 bank.
#
# Stock recipe, deliberately unchanged: quiet_weight 0.0, 800 steps, one validation, the
# standard 1.10 flow-ratio / 0.05 drift limits. This is the exact analog of the Pi0 first
# attempt (se_pi0_t2), which was REJECTED at these settings; matching it is the point.
#
# WHAT THIS RUN IS EVIDENCE FOR, AND WHAT IT IS NOT.
#   The guard limit 1.10 was calibrated on a flow-matching MSE (Pi0 base_quiet_flow_loss
#   0.00900). On Pi0-FAST, model.compute_loss returns TOKEN CROSS-ENTROPY -- the ratio is
#   still scale-free, but a 10% relative rise in cross-entropy is not the same amount of
#   forgetting as a 10% rise in flow MSE. Read the ratio, but read base_quiet_flow_loss
#   first: it is the number that says what scale we are on.
#   CONFOUND, recorded before the run: this bank carries 80 staged teacher deltas against
#   Pi0 329 on the same task, same 32 offsets, same 0.1087 threshold (85% of Pi0-FAST
#   records are NO_RISK vs 54% for Pi0). A rejection here therefore does NOT by itself
#   isolate the objective -- "too little teacher signal" stays live until the dose-matched
#   Pi0 control (Pi0 subsampled to 80 deltas) is run.
#
# LoRA surface differs structurally and this is the point of the arm: the Pi0 config names
# paligemma_variant=gemma_2b_lora AND action_expert_variant=gemma_300m_lora; the Pi0-FAST
# config has no action_expert_variant field at all, because there is no action expert.
#
# Training needs no pinned host (only collection/evaluation do). Equivalent 48 GB cards only
# -- SMALL_GPU_HOST-* has 11 GB and OOMs. Do NOT add "-l h=": it conflicts with -q and the job
# then sits in qw forever with no reason given.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OP=external/VLA-Arena/vla_arena/models/openpi
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python
STEPS=${STEPS:-800}
BATCH=${BATCH:-32}
QW=${QW:-0.0}
SRC=$LR/se_pi0fast_t2
ARM=PI0FAST_T2_q00
OUTA=$LR/se_pi0fast_t2/adapter/$ARM
BASE=${SE_VLA_ROOT}/checkpoints/pi0_fast_vla_arena_finetuned

f=$SRC/s1s2_records/derived/folds/offset_0/train.jsonl
[ -s "$f" ] || { echo "PI0FT_ABORT: no record set at $f"; exit 1; }
[ -d "$BASE/params" ] || { echo "PI0FT_ABORT: no params at $BASE"; exit 1; }

# The three published checkpoints nest their norm_stats DIFFERENTLY, and the trainer default
# only matches Pi0.5. Layouts, all verified on disk:
#   pi05      assets/VLA_Arena_L0_L_lerobot_openpi/VLA_Arena/   <- trainer DEFAULT_ASSET_ID
#   pi0       assets/VLA-Arena/VLA_Arena_L0_L_lerobot_openpi/   (hyphen)
#   pi0-FAST  assets/VLA_Arena/VLA_Arena_L0_L_lerobot_openpi/   (underscore)
# Without --asset-id the run dies AFTER model init with a norm_stats error. Assert first.
ASSET_ID=VLA_Arena/VLA_Arena_L0_L_lerobot_openpi
NS=$BASE/assets/$ASSET_ID/norm_stats.json
[ -s "$NS" ] || { echo "PI0FT_ABORT: no norm_stats at $NS"; exit 1; }
echo "    norm_stats OK: $NS"
echo "=== $ARM on $(hostname): rows=$(wc -l < "$f") steps=$STEPS batch=$BATCH quiet_weight=$QW"

export SE_VLA_TRAIN_BASE_CHECKPOINT="$BASE"
export SE_VLA_TRAIN_LORA_CONFIG=pi0_fast_vla_arena_low_mem_finetune

PYTHONPATH=src:external/VLA-Arena:. $VENV $OP/scripts/train_phase2_lora.py \
  --offset 0 \
  --records-root "$SRC/s1s2_records/blobs" \
  --derived-root "$SRC/s1s2_records/derived/folds" \
  --batch-size "$BATCH" --quiet-weight "$QW" \
  --assets-dir "$BASE/assets" --asset-id "$ASSET_ID" \
  --smoke-steps "$STEPS" --validation-interval "$STEPS" --patience 99 \
  --output-root "$OUTA"
rc=$?
echo "PI0FTRAIN_EXIT=$rc"

$VENV - "$OUTA" <<"PYV"
import json, pathlib, sys
d = pathlib.Path(sys.argv[1]) / "offset_0"
for name in ("metrics.json", "metrics_rejected.json"):
    p = d / name
    if p.exists():
        m = json.load(open(p))
        print("VERDICT file=%s accepted=%s" % (name, m.get("accepted")))
        print("  base_quiet_flow_loss = %s   <-- the scale the 1.10 limit is being applied to"
              % m.get("base_quiet_flow_loss"))
        print("  base_triggered_loss  = %s" % m.get("base_triggered_loss"))
        print("  best_quiet_flow_ratio= %s (limit %s)"
              % (m.get("best_quiet_flow_ratio"), m.get("quiet_flow_loss_ratio_limit")))
        print("  best_quiet_action_drift= %s (limit %s)"
              % (m.get("best_quiet_action_drift"), m.get("quiet_action_drift_limit")))
        print("  best_triggered_loss  = %s" % m.get("best_triggered_loss"))
        print("  validations_run=%s rejected_validations=%s"
              % (m.get("validations_run"), m.get("rejected_validations")))
        break
else:
    print("NO METRICS FILE in", d); sys.exit(1)
PYV
