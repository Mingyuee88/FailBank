#!/usr/bin/env bash
#$ -N R0b_fixed
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_D
#$ -pe smp 4
#$ -t 1-12
#$ -tc 2
#$ -o ${WORK_ROOT}/lora_dagger/sec_r0/batch_fixed/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/sec_r0/batch_fixed/j.$JOB_ID.$TASK_ID.out
#
# SEC R0b -- the four arms at a COMMON, FIXED step budget.
#
# Why this supersedes R0 (job 1372062/1372074). The adaptive early stop made the four arms
# incomparable, exactly the defect PREREG_2026-08-16 §1.2 records for R2 vs R2CTRL:
#
#     arm      best_step   base_triggered_loss   best_triggered_loss
#     GATED       25             0.1127               0.1134
#     MATCHED     50             0.1084               0.1061
#     SHAM        50             0.1084               0.1029
#
# GATED trained half as long as the others, so any difference carried a training-length
# term. Worse, each arm's stopping signal is computed on ITS OWN validation records, in
# which a DIFFERENT subset has had its delta zeroed -- so the three `base_triggered_loss`
# values are not the same quantity and the three arms were optimising different objectives.
#
# The registered fix (PREREG_2026-08-16 §2, rules S1 and S3): one validation set per
# comparison, and a common step budget with no arm stopping earlier for want of patience.
# Implemented here in the simplest form that cannot be gamed:
#
#   --smoke-steps N --validation-interval N --patience 99
#
# so every arm trains exactly N steps, validates exactly once, and checkpoint SELECTION is
# eliminated rather than merely equalised. The surviving guards (quiet action drift,
# quiet flow ratio) are computed on QUIET steps, which are byte-identical across all four
# arms -- deltas are only ever written on triggered steps -- so the guard is a shared
# scale by construction, satisfying S1 without needing a separate validation root.
#
# Two budgets, so the comparison reports a slope rather than one point.
#
# The primary endpoint is NOT Arena success. It is stage migration on the HELD-OUT records,
# measured offline with support_annotate.py -- which is what R0 actually asks (does gating
# make support expansion generalise?) and, per S4, does not look at rollouts at all.
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

ARMS=(GATED MATCHED SHAM ANTI)
ROOTS=(gated_records matched_records sham_gated_records anti_gated_records)
i=$(( (SGE_TASK_ID - 1) % 4 ))
case $(( (SGE_TASK_ID - 1) / 4 )) in
  0) STEPS=50 ;;
  1) STEPS=150 ;;
  2) STEPS=100 ;;
  *) echo "no budget for task $SGE_TASK_ID"; exit 0 ;;
esac
ARM=${ARMS[$i]}
ROOT=$SEC/${ROOTS[$i]}
HELDOUT=0
OUT=$SEC/train_fixed/${ARM}_s${STEPS}
mkdir -p "$OUT" "$SEC/batch_fixed"
if [ -s "$OUT/offset_0/metrics.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

# Source audit: every arm must carry exactly 484 residuals, and the arm must not be a
# duplicate of GATED unless it is meant to be.
external/VLA-Arena/envs/openpi/.venv/bin/python - "$ROOT" "$ARM" "$HELDOUT" "$SEC" <<'PYCHK'
import json, math, sys, pathlib
root, arm, heldout, SEC = sys.argv[1:5]
def residuals(p):
    out = {}
    for line in open(p):
        r = json.loads(line)
        nom, ex = r["nominal_action"], r["executed_action"]
        cl = [max(-1.0, min(1.0, float(x))) for x in nom[:3]]
        d = math.sqrt(sum((float(ex[i]) - cl[i]) ** 2 for i in range(3)))
        if d > 1e-9:
            out[r["record_id"]] = d
    return out
mine = residuals(pathlib.Path(root) / "derived/folds" / f"offset_{heldout}/train.jsonl")
gm = residuals(pathlib.Path(SEC) / "gated_records/derived/folds" / f"offset_{heldout}/train.jsonl")
tot, gt = sum(mine.values()), sum(gm.values())
print(f"ARM {arm}: residual_nonzero={len(mine)} total_magnitude={tot:.4f}")
assert len(mine) == 484, f"{arm} must carry exactly 484 residuals, found {len(mine)}"
ov = len(set(mine) & set(gm))
print(f"  overlap with GATED: {ov}/484 ({100*ov/484:.1f}%)  magnitude vs GATED: {tot:.4f}/{gt:.4f}")
if arm == "GATED":
    assert ov == 484 and abs(tot - gt) < 1e-9, "GATED is not itself"
elif arm == "SHAM":
    assert ov == 484 and abs(tot - gt) < 1e-6, "SHAM must be GATED's records at GATED's magnitudes"
elif arm == "ANTI":
    assert ov < 484 and tot > gt, "ANTI must be a different, larger-magnitude set"
elif arm == "MATCHED":
    assert ov < 484, "MATCHED is identical to GATED -- the control is void"
print("R0B_ROOT_VERIFIED")
PYCHK

echo "=== R0b arm=$ARM steps=$STEPS root=$ROOT on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  $OP/scripts/train_phase2_lora.py \
  --offset "$HELDOUT" \
  --records-root "$ROOT/blobs" \
  --derived-root "$ROOT/derived/folds" \
  --batch-size 2 \
  --quiet-weight 0.0 \
  --smoke-steps "$STEPS" \
  --validation-interval "$STEPS" \
  --patience 99 \
  --output-root "$OUT"
rc=$?
echo "R0B_TRAIN_EXIT=$rc arm=$ARM steps=$STEPS"

# Effective-value check: the budget must be the one requested, not the default.
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$STEPS" "$ARM" <<'PYV'
import json, sys, pathlib
out, steps, arm = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = json.load(open(pathlib.Path(out) / "offset_0/metrics.json"))
print(f"{arm}: best_step={m.get('best_step')} validations_run={m.get('validations_run')} "
      f"stopped_early={m.get('stopped_early')} accepted={m.get('accepted')}")
assert m.get("best_step") == steps, \
    f"budget not honoured: best_step={m.get('best_step')} but {steps} was requested"
assert m.get("validations_run") == 1, \
    f"more than one validation ran ({m.get('validations_run')}) -- selection was not eliminated"
print("R0B_BUDGET_VERIFIED")
PYV
