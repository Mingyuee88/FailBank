#!/usr/bin/env bash
#$ -N v27_train
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_D
#$ -pe smp 2
#$ -t 1-3
#$ -tc 1
#$ -o ${WORK_ROOT}/lora_dagger/v27_ablate/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v27_ablate/batch/j.$JOB_ID.$TASK_ID.out
#
# v27_ablate -- the attribution experiment the whole claim rests on, and which has been
# missing since the beginning.
#
# THE PROBLEM. Arm A (shield corrections distilled into LoRA) takes transfer SR from
# 116/150 to 121/150. Nothing in the record separates three explanations:
#
#   (1) the shield's corrections carry useful safety information          <- the claim
#   (2) touching the LoRA weights at all on this state distribution helps
#   (3) any perturbation of matched magnitude helps
#
# Explanation (3) is not hypothetical here. The permutation control that this project
# ran on its external-memory line found that shuffling the key->residual pairing did not
# degrade performance at all -- the shuffled arm was the best-CC arm in the whole study.
# The memory's CONTENT contributed nothing. Until the same control is run on the LoRA
# line, there is no reason to believe this line is different.
#
# WHAT MAKES A CLEAN CONTROL POSSIBLE. The teacher signal decomposes exactly:
#
#     executed_action = clip(nominal_action) + projection_delta
#
# and `projection_delta` is nonzero on precisely the records whose
# `formal_projection.triggered` is true -- checked count for count (866/2830 in this
# fold). So the shield's contribution can be deleted or scrambled while holding
# everything else identical: same episodes, same states, same observation blobs (they are
# symlinked, not copied), same record count, same quiet flags, same optimizer, same
# batch size, same step count, same host.
#
#   SFT0    projection_delta := 0. The target becomes the base policy's own clipped
#           action, i.e. pure self-distillation. Isolates explanation (2).
#   SHAM    projection_delta := random direction, SAME magnitude, per record, applied
#           only in the 3-D translation subspace the CBF ever writes to. Isolates
#           explanation (3). Verified at build time: magnitude error < 1e-15 and the
#           original direction is reproduced on 1 record in 2830 (chance, expected 0.2).
#   SFTPOS  behaviour cloning on the base policy's own SUCCESSFUL episodes, corrections
#           removed. A different null from the other two: it asks whether ordinary
#           imitation of what already works matches distilled shield feedback.
#
# READING THE RESULT IN ADVANCE, so it cannot be reinterpreted afterwards:
#   - if SHAM matches arm A, the shield's CONTENT contributes nothing and the honest
#     description of this method is a magnitude-scheduled perturbation, not distilled
#     safety knowledge. That is a publishable negative result and it is what will be
#     written.
#   - if SFT0 matches arm A, the gain is generic fine-tuning and the shield is irrelevant.
#   - only if arm A beats BOTH nulls on the untouched transfer set is the feedback claim
#     supported.
#
# LEAKAGE. SFT0 and SHAM inherit arm A's records root, which contains only base-FAILURE
# offsets -- none of them in the held-out retention probe. SFTPOS draws from base-SUCCESS
# offsets, so the 18 held-out ones were dropped at build time (43389 records excluded),
# keeping every arm's training set disjoint from the same probe.
#
# HOST. Pinned to the host arm A was trained on. GPU model changes numerics, and a
# four-way comparison must not carry that term.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OP=external/VLA-Arena/vla_arena/models/openpi

case "$SGE_TASK_ID" in
  1) ARM=SFT0;   ROOT=$LR/v27_ablate/sft0_records;   HELDOUT=0; EXPECT=none ;;
  2) ARM=SHAM;   ROOT=$LR/v27_ablate/sham_records;   HELDOUT=0; EXPECT=matched ;;
  # SFTPOS draws from base-SUCCESS offsets, so it has no offset_0 fold. Its LOSO fold
  # must be held out from a TRAINING-half offset: the 18 retention-probe offsets were
  # dropped from every split at build time, so a fold named after one of them has an
  # empty validation split and the trainer -- correctly -- refuses it. Offset 1 is in the
  # training half, so fold offset_1 trains on the remaining training-half successes and
  # validates on offset 1, touching the retention probe nowhere.
  # Consequence recorded: the trainer derives its data-order seed from --offset, so
  # SFTPOS does not share the other arms' data order. It is a different-data null, not a
  # matched-target one, so this does not confound the comparison it is there to make.
  3) ARM=SFTPOS; ROOT=$LR/v27_ablate/sftpos_records; HELDOUT=1; EXPECT=none ;;
esac

OUT=$LR/v27_ablate/train/$ARM
mkdir -p "$OUT" $LR/v27_ablate/batch
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

# Source audit before spending an hour of GPU. The specific hazard is producing an
# ablation root that is byte-identical to arm A's, which would make the null trivially
# "match" the method and look like a decisive negative result. Assert the residuals are
# what this arm claims they are, measured off the fold the trainer will actually read.
external/VLA-Arena/envs/openpi/.venv/bin/python - "$ROOT" "$ARM" "$HELDOUT" "$EXPECT" "$LR" <<'PYCHK'
import json, math, sys, pathlib
root, arm, heldout, expect, LR = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
f = pathlib.Path(root) / "derived" / "folds" / f"offset_{heldout}" / "train.jsonl"
assert f.exists(), f"missing fold {f}"
n = touched = 0
norms = []
for line in open(f):
    r = json.loads(line)
    nom, ex = r["nominal_action"], r["executed_action"]
    cl = [max(-1.0, min(1.0, float(x))) for x in nom[:3]]
    d = math.sqrt(sum((float(ex[i]) - cl[i]) ** 2 for i in range(3)))
    n += 1
    if d > 1e-9:
        touched += 1
        norms.append(d)
mean = sum(norms) / len(norms) if norms else 0.0
print(f"ARM {arm}: fold=offset_{heldout} records={n} residual_nonzero={touched} mean_residual={mean:.4f}")
if expect == "none":
    assert touched == 0, f"{arm} must carry no shield residual, found {touched}"
else:
    # magnitudes must match arm A's source fold exactly, only the directions differ
    ref = pathlib.Path(LR) / "phase1_collection/records/derived/folds" / f"offset_{heldout}" / "train.jsonl"
    rn = []
    for line in open(ref):
        r = json.loads(line)
        nom, ex = r["nominal_action"], r["executed_action"]
        cl = [max(-1.0, min(1.0, float(x))) for x in nom[:3]]
        d = math.sqrt(sum((float(ex[i]) - cl[i]) ** 2 for i in range(3)))
        if d > 1e-9:
            rn.append(d)
    assert len(rn) == len(norms), f"residual count {len(norms)} != arm A's {len(rn)}"
    assert abs(sum(rn) - sum(norms)) < 1e-6, "sham did not preserve total residual magnitude"
    print(f"  magnitude-matched to arm A: {len(rn)} residuals, total {sum(rn):.4f}")
print("ABLATION_ROOT_VERIFIED")
PYCHK

echo "=== v27 arm=$ARM root=$ROOT heldout=$HELDOUT on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  $OP/scripts/train_phase2_lora.py \
  --offset "$HELDOUT" \
  --records-root "$ROOT/blobs" \
  --derived-root "$ROOT/derived/folds" \
  --batch-size 2 \
  --quiet-weight 0.0 \
  --output-root "$OUT"
echo "TRAIN_EXIT=$? arm=$ARM"
