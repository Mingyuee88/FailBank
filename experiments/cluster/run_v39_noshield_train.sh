#!/usr/bin/env bash
#$ -N v39_nsht
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_D
#$ -pe smp 2
#$ -t 1-1
#$ -tc 1
#$ -o ${WORK_ROOT}/lora_dagger/v39_noshield_train/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v39_noshield_train/batch/j.$JOB_ID.$TASK_ID.out
#
# v39 -- train the shield-off control, the arm that decides what the method's active
# ingredient is.
#
# The comparison it completes:
#
#   SFT0      states from episodes the shield was steering, targets = the policy's own
#             clipped actions.        L2 cost -41.4% (Wilcoxon p=0.009)
#   SFTPOS    states from episodes the shield almost never fired in, same target rule.
#                                     L2 cost -21.6% (p=0.63)
#   NOSHIELD  states from episodes with no shield at all, same target rule, and the SAME
#             ten offsets SFT0 used.  <- this job
#
# Every one of the three uses the base policy's own actions as targets, so none of them
# carries any corrective signal. If NOSHIELD is much weaker than SFT0, what the runtime
# shield contributes is the state distribution it drags the policy through, and it earns
# its place as an exploration device even though its corrections do not survive their own
# ablation. If NOSHIELD matches SFT0, this line is ordinary self-distillation and the
# shield contributes nothing; that is then the result.
#
# Fold offset_0, matching SFT0's fold, so the two differ in collection and nothing else.
# Record counts are printed rather than assumed equal: shield-off episodes run to a
# different length, and if the two roots differ substantially in size that is a confound
# to report, not to hide.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OP=external/VLA-Arena/vla_arena/models/openpi

ARM=NOSHIELD
ROOT=$LR/v38_noshield/records
HELDOUT=0
OUT=$LR/v39_noshield_train/$ARM
mkdir -p "$OUT" $LR/v39_noshield_train/batch
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

external/VLA-Arena/envs/openpi/.venv/bin/python - "$ROOT" "$LR" "$HELDOUT" <<'PYCHK'
import json, math, pathlib, sys
root, LR, heldout = sys.argv[1], sys.argv[2], sys.argv[3]
f = pathlib.Path(root) / "derived/folds" / f"offset_{heldout}" / "train.jsonl"
ref = pathlib.Path(LR) / "v27_ablate/sft0_records/derived/folds" / f"offset_{heldout}" / "train.jsonl"
assert f.exists(), f"missing fold {f}"

def stats(p):
    n = touched = 0
    for line in open(p):
        r = json.loads(line)
        nom, ex = r["nominal_action"], r["executed_action"]
        cl = [max(-1.0, min(1.0, float(x))) for x in nom[:3]]
        d = math.sqrt(sum((float(ex[i]) - cl[i]) ** 2 for i in range(3)))
        n += 1
        if d > 1e-9:
            touched += 1
    return n, touched

n, t = stats(f)
print(f"NOSHIELD fold=offset_{heldout} records={n} residual_nonzero={t}")
assert t == 0, f"the shield-off control carries {t} nonzero residuals -- it is not shield-off"
if ref.exists():
    rn, rt = stats(ref)
    print(f"  SFT0 comparison root: records={rn} residual_nonzero={rt} "
          f"(size ratio noshield/sft0 = {n / max(rn, 1):.2f})")
print("NOSHIELD_ROOT_VERIFIED")
PYCHK

echo "=== v39 arm=$ARM root=$ROOT heldout=$HELDOUT on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  $OP/scripts/train_phase2_lora.py \
  --offset "$HELDOUT" \
  --records-root "$ROOT/blobs" \
  --derived-root "$ROOT/derived/folds" \
  --batch-size 2 \
  --quiet-weight 0.0 \
  --output-root "$OUT"
echo "TRAIN_EXIT=$? arm=$ARM"
