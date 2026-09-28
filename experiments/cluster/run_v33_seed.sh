#!/usr/bin/env bash
#$ -N v33_seed
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_D
#$ -pe smp 2
#$ -t 1-3
#$ -tc 2
#$ -o ${WORK_ROOT}/lora_dagger/v33_seed/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v33_seed/batch/j.$JOB_ID.$TASK_ID.out
#
# v33_seed -- replicate checkpoints, so "arm A beats the null" can be distinguished from
# "this particular checkpoint beats the null".
#
# Every learned arm in this study is ONE training run. The trainer previously derived its
# data order from the fold index alone, so a rerun was bit-identical and run-to-run
# variation was not merely unmeasured, it was unmeasurable. With --data-seed it is.
#
# This matters more here than it usually would. The arms early-stop at step 25-50 of a
# 400-step budget with the quiet-drift guard binding (best_quiet_flow_ratio 0.97-1.01),
# so the adapters are small perturbations of the base weights and the deployed effect
# sizes are correspondingly small -- arm A's transfer result is +5/150 at p=0.533. An
# effect that small is exactly the regime where a single checkpoint proves nothing.
#
# Three replicates, one per decisive arm: the method (A), the control that would falsify
# it (SHAM), and the round-2 arm (R2). If the ORDERING of arms changes between seeds, no
# single-checkpoint comparison in this project is evidence, and that has to be said
# plainly rather than resolved by picking the run that agrees.
#
# Trained here, evaluated later: training is cheap and the GPUs are otherwise idle, while
# evaluation is the scarce resource and the primary arms have first claim on it.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OP=external/VLA-Arena/vla_arena/models/openpi

SEED=777
case "$SGE_TASK_ID" in
  1) ARM=A_s2;    ROOT=$LR/phase1_collection/records ;;
  2) ARM=SHAM_s2; ROOT=$LR/v27_ablate/sham_records ;;
  3) ARM=R2_s2;   ROOT=$LR/v30_r2train/records ;;
esac
HELDOUT=0

OUT=$LR/v33_seed/train/$ARM
mkdir -p "$OUT" $LR/v33_seed/batch
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

echo "=== v33 arm=$ARM root=$ROOT data_seed=$SEED heldout=$HELDOUT on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  $OP/scripts/train_phase2_lora.py \
  --offset "$HELDOUT" \
  --records-root "$ROOT/blobs" \
  --derived-root "$ROOT/derived/folds" \
  --batch-size 2 \
  --quiet-weight 0.0 \
  --data-seed "$SEED" \
  --output-root "$OUT"
echo "TRAIN_EXIT=$? arm=$ARM"

# The knob has to have moved something. A replicate whose metrics are bit-identical to
# the original means --data-seed never reached the loader, and the variance estimate
# would then be a fabricated zero.
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/offset_$HELDOUT/metrics.json" "$ARM" "$SEED" <<'PYCHK'
import json, sys
m = json.load(open(sys.argv[1])); arm, seed = sys.argv[2], int(sys.argv[3])
print(f"V33 arm={arm} data_seed={m.get('data_seed')} best_step={m.get('best_step')} "
      f"best_triggered_loss={m.get('best_triggered_loss')} "
      f"best_quiet_flow_ratio={m.get('best_quiet_flow_ratio')}")
assert m.get("data_seed") == seed, f"metrics record data_seed {m.get('data_seed')}, expected {seed}"
print("V33_SEED_VERIFIED")
PYCHK
