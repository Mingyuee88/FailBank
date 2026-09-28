#!/usr/bin/env bash
#$ -N v30_r2tr
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_D
#$ -pe smp 2
#$ -t 1-2
#$ -tc 1
#$ -o ${WORK_ROOT}/lora_dagger/v30_r2train/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v30_r2train/batch/j.$JOB_ID.$TASK_ID.out
#
# v30_r2train -- round 2 of the loop, and the control that decides whether the loop is
# real.
#
# ROUND 2 (arm R2). Trained from BASE weights on round-1 records aggregated with the
# round-2 records collected on R1's OWN states (v26_r2collect). Base init rather than
# warm-starting from R1: aggregation is what DAgger actually specifies, warm-starting
# would risk silent forgetting of round-1 data, and confounding "collected on the evolved
# policy" with "had a second pass of optimization". Warm-start is available as a later
# ablation, not as the headline.
#
# What round 2 saw that round 1 could not. R1's failure set is not base's:
#   inherited {0,10,26,29,36,42} -- base failures R1 did not fix
#   NEW       {17,34,37}         -- successes R1 broke itself, episodes that did not
#                                   exist until R1 existed
# and the collection confirms the teacher still has signal on that distribution: running
# the shield on R1 repaired all 7 of R1's failures inside the training half, while
# breaking 3 of its 22 successes (13.6%) -- against 9 of 36 (25%) when the same shield
# was run on base. The teacher interferes less with the evolved policy than with the one
# it was built for, which is the first quantitative sign that the loop is doing something.
#
# THE CONTROL (arm R2CTRL) IS THE POINT. Round-2 training differs from round-1 training
# in TWO ways at once: whose distribution the new records came from, and how many records
# there are (2830 -> 6612 in the fold). A gain is equally well explained by ordinary data
# scaling. R2CTRL aggregates round-1 records with records collected by the SAME shield on
# the SAME 29 offsets with the SAME recorder on the SAME host -- but on BASE's states
# (v29_basedist). Same budget, same everything, one variable.
#
#   R2 > R2CTRL   the gain is attributable to collecting on the evolved policy's own
#                 distribution. This is the self-evolving claim.
#   R2 = R2CTRL   the second round bought nothing a second pass over base data would not
#                 have bought. The loop is not doing what the paper would claim, and the
#                 method should be described as a single offline update trained on more
#                 data. That will be written plainly if it is what comes out.
#
# HOST. Pinned to the host arm A (=R1) trained on. GPU model changes numerics; a
# round-over-round comparison must not carry that term.
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
  1) ARM=R2;     ROOT=$LR/v30_r2train/records;      SRC=v26_r2collect ;;
  2) ARM=R2CTRL; ROOT=$LR/v30_r2ctrl/records;       SRC=v29_basedist ;;
esac
HELDOUT=0

OUT=$LR/v30_r2train/train/$ARM
mkdir -p "$OUT" $LR/v30_r2train/batch
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

# Source audit. The hazard: a merged root that silently failed to pick up the second
# half, leaving R2 identical to R1 and producing a "no gain from round 2" null that is
# really a missing-data bug. Assert the fold is strictly larger than round 1's and that
# it carries records from episodes round 1 could not have contained.
external/VLA-Arena/envs/openpi/.venv/bin/python - "$ROOT" "$ARM" "$HELDOUT" "$LR" <<'PYCHK'
import json, pathlib, sys, collections
root, arm, heldout, LR = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
f = pathlib.Path(root) / "derived/folds" / f"offset_{heldout}" / "train.jsonl"
ref = pathlib.Path(LR) / "phase1_collection/records/derived/folds" / f"offset_{heldout}" / "train.jsonl"
assert f.exists(), f"missing fold {f}"
def read(p):
    ids, offs, trig = set(), collections.Counter(), 0
    for line in open(p):
        r = json.loads(line)
        ids.add(r["record_id"]); offs[str(r["offset"])] += 1
        if (r.get("formal_projection") or {}).get("triggered"):
            trig += 1
    return ids, offs, trig
nid, noff, ntrig = read(f)
rid, roff, rtrig = read(ref)
new = nid - rid
print(f"ARM {arm}: fold=offset_{heldout} records={len(nid)} triggered={ntrig} "
      f"round1_records={len(rid)} new_records={len(new)}")
print(f"  round1 offsets: {sorted(roff, key=int)}")
print(f"  merged offsets: {sorted(noff, key=int)}")
assert len(new) > 0, "merged fold contains nothing round 1 did not already have"
assert rid <= nid, "merged fold dropped round-1 records -- aggregation is not aggregation"
print("R2_ROOT_VERIFIED")
PYCHK

echo "=== v30 arm=$ARM root=$ROOT src=$SRC heldout=$HELDOUT on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  $OP/scripts/train_phase2_lora.py \
  --offset "$HELDOUT" \
  --records-root "$ROOT/blobs" \
  --derived-root "$ROOT/derived/folds" \
  --batch-size 2 \
  --quiet-weight 0.0 \
  --output-root "$OUT"
echo "TRAIN_EXIT=$? arm=$ARM"
