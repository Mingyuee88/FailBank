#!/usr/bin/env bash
#$ -N R0_train
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_D
#$ -pe smp 4
#$ -t 1-4
#$ -tc 2
#$ -o ${WORK_ROOT}/lora_dagger/sec_r0/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/sec_r0/batch/j.$JOB_ID.$TASK_ID.out
#
# SEC R0 -- does support-gating the training targets do what training on everything did not?
#
# D was trained on all 866 outcome-filtered shield corrections. Offline annotation (E12)
# shows only 484 of them (55.9%) lie inside base's own action support. E12/E13 then showed
# that D's support DID move toward the corrections it was trained on in-sample (high bucket
# 20 release / 0 regression) but that this does NOT generalise to held-out records
# (high bucket 2/2, no movement; med+high pooled 6 vs 2, p = 0.289).
#
# R0 tests the one remaining untested explanation: that mixing in the 382 corrections the
# policy cannot represent is what prevents support expansion from generalising.
#
# Three arms, all dose-matched by construction -- identical records, episodes, observations,
# quiet flags, record count and optimizer steps. Deltas are ZEROED rather than records
# dropped, so nothing but the teacher signal differs:
#
#   GATED    delta kept on the 484 in-support records          the treatment
#   MATCHED  delta kept on 484 records drawn at random WITHIN   separates "gating helps"
#            each teacher bucket, so bucket composition is      from "less teacher signal
#            identical to GATED                                 helps"
#   SHAM     the 484 in-support records keep their magnitude    separates the CONTENT of
#            but get a random direction                         in-support corrections from
#                                                               their dose and placement
#   ANTI     delta kept on the 484 records FURTHEST outside     the other end of the same
#            support, ranked WITHIN each bucket so composition  ranking; maximum contrast
#            still matches gated exactly                        on the support axis alone
#
# All four arms carry bucket composition low=60 med=109 high=315, so nothing here can be
# explained by the outcome filter's own preferences. Support-ratio medians run
# 0.281 (gated) < 0.505 (matched) < 1.539 (anti), monotone by construction.
#
# ⚠ ONE CONFOUND, stated in advance. Corrections further outside support are also LARGER:
# total residual magnitude is 88.83 (gated) against 110.20 (anti), +24%. Support distance
# and magnitude cannot both be matched. The reading is therefore asymmetric and is fixed
# here, before the result:
#   - ANTI significantly worse  -> consistent with the support axis, but confounded with
#                                  magnitude; a magnitude-matched arm would be required.
#   - ANTI not worse            -> the support axis does not carry, and this conclusion is
#                                  STRONGER for the confound, since even a 24% magnitude
#                                  difference produced nothing.
# GATED vs SHAM is unaffected: identical records, identical magnitudes, direction only.
#
# SHAM is preregistered, not added afterwards: the same control killed this project's
# external-memory line on 2026-08-11 (shuffled pairing, dose held fixed, McNemar p = 1.000).
#
# HOST. Pinned to HOST_D, the host the v27 ablation arms were trained on, so the three
# arms carry no GPU-model term. D's own training host is not recorded, so GATED-vs-D is
# reported with that caveat; GATED-vs-MATCHED-vs-SHAM is the comparison that matters and it
# is internally pinned.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OP=external/VLA-Arena/vla_arena/models/openpi
SEC=$LR/sec_r0

case "$SGE_TASK_ID" in
  1) ARM=GATED;  ROOT=$SEC/gated_records;      EXPECT=real ;;
  2) ARM=MATCHED; ROOT=$SEC/matched_records;   EXPECT=real ;;
  3) ARM=SHAM;   ROOT=$SEC/sham_gated_records; EXPECT=sham ;;
  4) ARM=ANTI;   ROOT=$SEC/anti_gated_records; EXPECT=real ;;
  *) echo "no arm for task $SGE_TASK_ID"; exit 0 ;;
esac
HELDOUT=0
OUT=$SEC/train/$ARM
mkdir -p "$OUT" "$SEC/batch"
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

# Source audit before spending GPU. The hazard is an arm whose fold is byte-identical to
# another's, which would make the control trivially "match" the treatment.
external/VLA-Arena/envs/openpi/.venv/bin/python - "$ROOT" "$ARM" "$HELDOUT" "$EXPECT" "$SEC" <<'PYCHK'
import json, math, sys, pathlib
root, arm, heldout, expect, SEC = sys.argv[1:6]
def residuals(p):
    out = []
    for line in open(p):
        r = json.loads(line)
        nom, ex = r["nominal_action"], r["executed_action"]
        cl = [max(-1.0, min(1.0, float(x))) for x in nom[:3]]
        d = math.sqrt(sum((float(ex[i]) - cl[i]) ** 2 for i in range(3)))
        if d > 1e-9:
            out.append((r["record_id"], d))
    return out
f = pathlib.Path(root) / "derived" / "folds" / f"offset_{heldout}" / "train.jsonl"
assert f.exists(), f"missing fold {f}"
mine = residuals(f)
tot = sum(d for _, d in mine)
print(f"ARM {arm}: fold=offset_{heldout} residual_nonzero={len(mine)} total_magnitude={tot:.4f}")
assert len(mine) == 484, f"{arm} must carry exactly 484 residuals, found {len(mine)}"
g = residuals(pathlib.Path(SEC) / "gated_records/derived/folds" / f"offset_{heldout}/train.jsonl")
gm = {rid: d for rid, d in g}
if expect == "sham":
    # same records, same magnitudes, different directions
    assert set(gm) == {rid for rid, _ in mine}, "sham is not on the gated record set"
    assert abs(sum(gm.values()) - tot) < 1e-6, "sham did not preserve total magnitude"
    print(f"  magnitude-matched to GATED on the same {len(mine)} records")
elif arm in ("MATCHED", "ANTI"):
    ov = len(set(gm) & {rid for rid, _ in mine})
    assert ov < len(mine), f"{arm} selected exactly the GATED set -- the control is void"
    print(f"  overlap with GATED: {ov}/{len(mine)} records ({100*ov/len(mine):.1f}%)")
    if arm == "ANTI":
        # effective-value check: this arm claims the records FURTHEST outside support,
        # so its residuals must be larger in total than GATED's, not smaller
        gt = sum(gm.values())
        assert tot > gt, (f"ANTI total magnitude {tot:.4f} <= GATED's {gt:.4f}; the "
                          "selection sign is inverted")
        print(f"  total magnitude {tot:.4f} vs GATED {gt:.4f} (+{100*(tot-gt)/gt:.1f}%)")
print("R0_ROOT_VERIFIED")
PYCHK

echo "=== R0 arm=$ARM root=$ROOT heldout=$HELDOUT on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  $OP/scripts/train_phase2_lora.py \
  --offset "$HELDOUT" \
  --records-root "$ROOT/blobs" \
  --derived-root "$ROOT/derived/folds" \
  --batch-size 2 \
  --quiet-weight 0.0 \
  --output-root "$OUT"
echo "R0_TRAIN_EXIT=$? arm=$ARM"
