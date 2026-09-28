#!/usr/bin/env bash
#$ -N C_sweep
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 4
#$ -t 1-8
#$ -tc 2
#$ -o ${WORK_ROOT}/lora_dagger/c_sweep/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/c_sweep/batch/j.$JOB_ID.$TASK_ID.out
#
# C -- is the learning channel physically open at all?
#
# Every training arm in this project reported weights moving (act_drift 0.001-0.005 against a
# 0.05 limit, flow_ratio 0.80-1.02) while the behaviour distribution never moved: pi_0, pi_1
# and pi_2 all landed inside one distribution (crash 11.0 / 11.0 / 12.2, noise band +-5.5).
#
# The likely reason, found by reading the trainer's defaults:
#     config default   batch_size=32   num_train_steps=400   -> 12800 samples
#     what we ran      --batch-size 2  --smoke-steps 100     ->   200 samples   (1/64)
# and `--smoke-steps` overrides num_train_steps WITHOUT rescaling the cosine schedule, whose
# decay_steps stays 400 -- so at step 100 the run has barely cleared its 20-step warmup. The
# flag is documented as "for a quick smoke run"; it was used as the training recipe.
#
# This sweep varies only training strength on one fixed record set from one fixed checkpoint.
# Primary readout is act_drift, which measures how far the policy's actions moved on quiet
# steps -- free, continuous, and unaffected by the outcome fragility that makes crash counts
# uninformative at n=1.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OP=external/VLA-Arena/vla_arena/models/openpi
SEC=$LR/c_sweep
ROOT=$LR/se_r2/s1s2_records
BASE=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
STEPSET=(100 400 100 400 100 400 1000 1000)
BATCHSET=(2 2 8 8 32 32 8 32)
i=$((SGE_TASK_ID - 1))
S=${STEPSET[$i]}; B=${BATCHSET[$i]}
OUT=$SEC/s${S}_b${B}
mkdir -p "$OUT" "$SEC/batch"
if [ -s "$OUT/offset_0/metrics.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
echo "=== C sweep steps=$S batch=$B  (samples=$((S*B)); project default 400x32=12800) ==="
set +e
SE_VLA_TRAIN_BASE_CHECKPOINT="$BASE" PYTHONPATH=src:external/VLA-Arena:. \
  external/VLA-Arena/envs/openpi/.venv/bin/python \
  $OP/scripts/train_phase2_lora.py \
  --offset 0 --records-root "$ROOT/blobs" --derived-root "$ROOT/derived/folds" \
  --batch-size "$B" --quiet-weight 0.0 \
  --smoke-steps "$S" --validation-interval "$S" --patience 99 \
  --output-root "$OUT"
rc=$?
set -e
echo "C_EXIT=$rc steps=$S batch=$B"
if [ ! -s "$OUT/offset_0/metrics.json" ]; then
  echo "C_RESULT steps=$S batch=$B samples=$((S*B)) STATUS=failed_or_guard_rejected"
  exit 0
fi
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$S" "$B" <<'PYV'
import json, sys, pathlib
out, s, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m = json.load(open(pathlib.Path(out) / "offset_0/metrics.json"))
print(f"C_RESULT steps={s} batch={b} samples={s*b} "
      f"act_drift={m.get('best_quiet_action_drift'):.6f} "
      f"flow_ratio={m.get('best_quiet_flow_ratio'):.5f} "
      f"trig_loss={m.get('base_triggered_loss'):.5f}->{m.get('best_triggered_loss'):.5f} "
      f"accepted={m.get('accepted')}")
PYV
