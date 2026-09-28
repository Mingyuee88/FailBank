#!/usr/bin/env bash
#$ -N R1_train
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 4
#$ -t 1-8
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/c1_r1/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/c1_r1/batch/j.$JOB_ID.$TASK_ID.out
#
# R1 -- does staging the failure bank by lead time beat not staging it?
#
# Four dose-matched arms over the C1 failure collection (6101 records, 1800 of them from
# episodes that actually crashed -- the signal the old bank had zero of):
#
#   S1ONLY   225 deltas, all from S1_early (lead > 30 steps before the first crossing of
#            cost_pair_min_distance < 0.1087). The treatment.
#   MATCHED  225 deltas drawn at random within teacher bucket. Realised composition
#            S1 40 / NO_RISK 142 / POST 25 / S2 18 -- i.e. mostly NOT early. Isolates
#            "staging by lead time" from "less teacher signal".
#   SHAM     the same 225 S1_early records, each delta replaced by a random direction of
#            identical magnitude. Isolates CONTENT from dose and placement. This is the
#            control that killed this project's external-memory line (2026-08-11,
#            McNemar p = 1.000); it runs alongside, not afterwards.
#   ALL      all 2182 deltas -- "use the failure data but do not stage it", the existing
#            recipe on the new bank, and what the curriculum has to beat. Its dose is ~9.7x
#            the others by construction: that IS the curriculum's claim, not a flaw, but it
#            means only S1ONLY-vs-MATCHED isolates staging.
#
# Fixed budget, per PREREG_2026-08-16 S1/S3 and the R0 correction: --smoke-steps N with
# --validation-interval N and --patience 99, so every arm trains exactly N steps, validates
# once, and checkpoint SELECTION IS ELIMINATED rather than merely equalised. Under adaptive
# stopping the R0 arms landed on 25/50/50 steps with three mutually incomparable
# `base_triggered_loss` scales -- the same defect that voided R2 vs R2CTRL.
#
# Two budgets, because R0 showed the usable window is narrow: at 50 steps the policy barely
# moved, and by 100 steps the most in-support arm already breached the registered
# quiet-flow-ratio guard. If that repeats here the result is a null about the RECIPE, not
# about the curriculum, and is reported as such.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OP=external/VLA-Arena/vla_arena/models/openpi
SEC=$LR/c1_r1

ARMS=(S1ONLY MATCHED SHAM ALL)
ROOTS=(s1_only_records matched_records sham_records all_records)
EXPECT=(225 225 225 2182)
i=$(( (SGE_TASK_ID - 1) % 4 ))
case $(( (SGE_TASK_ID - 1) / 4 )) in
  0) STEPS=50 ;;
  1) STEPS=100 ;;
  *) echo "no budget for task $SGE_TASK_ID"; exit 0 ;;
esac
ARM=${ARMS[$i]}; ROOT=$SEC/${ROOTS[$i]}; WANT=${EXPECT[$i]}
OUT=$SEC/train/${ARM}_s${STEPS}
mkdir -p "$OUT" "$SEC/batch"
if [ -s "$OUT/offset_0/metrics.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

external/VLA-Arena/envs/openpi/.venv/bin/python - "$ROOT" "$ARM" "$WANT" "$SEC" <<'PYCHK'
import json, math, sys, pathlib, collections
root, arm, want, SEC = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
def residuals(p):
    out = {}
    for line in open(p):
        r = json.loads(line)
        nom, ex = r["nominal_action"], r["executed_action"]
        cl = [max(-1.0, min(1.0, float(x))) for x in nom[:3]]
        d = math.sqrt(sum((float(ex[i]) - cl[i]) ** 2 for i in range(3)))
        if d > 1e-9:
            out[r["record_id"]] = (d, r.get("curriculum_stage"))
    return out
mine = residuals(pathlib.Path(root) / "derived/folds/offset_0/train.jsonl")
st = collections.Counter(s for _, s in mine.values())
tot = sum(d for d, _ in mine.values())
print(f"ARM {arm}: residual_nonzero={len(mine)} total_magnitude={tot:.4f} stages={dict(st)}")
assert len(mine) == want, f"{arm} must carry exactly {want} residuals, found {len(mine)}"
if arm in ("S1ONLY", "SHAM"):
    assert set(st) == {"S1_early"}, f"{arm} leaked non-S1 stages: {dict(st)}"
if arm == "MATCHED":
    assert st.get("S1_early", 0) < want, "MATCHED is all-S1 -- the control is void"
if arm == "ALL":
    assert len(st) > 1, "ALL carries a single stage"
gm = residuals(pathlib.Path(SEC) / "s1_only_records/derived/folds/offset_0/train.jsonl")
if arm == "SHAM":
    gt = sum(d for d, _ in gm.values())
    assert abs(tot - gt) < 1e-6, f"SHAM magnitude {tot} != S1ONLY {gt}"
    print(f"  magnitude matches S1ONLY exactly: {tot:.4f}")
print("R1_ROOT_VERIFIED")
PYCHK

echo "=== R1 arm=$ARM steps=$STEPS root=$ROOT on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  $OP/scripts/train_phase2_lora.py \
  --offset 0 \
  --records-root "$ROOT/blobs" \
  --derived-root "$ROOT/derived/folds" \
  --batch-size 2 \
  --quiet-weight 0.0 \
  --smoke-steps "$STEPS" \
  --validation-interval "$STEPS" \
  --patience 99 \
  --output-root "$OUT"
rc=$?
echo "R1_TRAIN_EXIT=$rc arm=$ARM steps=$STEPS"

external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$STEPS" "$ARM" <<'PYV'
import json, sys, pathlib
out, steps, arm = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = json.load(open(pathlib.Path(out) / "offset_0/metrics.json"))
print(f"{arm}: best_step={m.get('best_step')} validations_run={m.get('validations_run')} "
      f"flow_ratio={m.get('best_quiet_flow_ratio'):.5f} "
      f"act_drift={m.get('best_quiet_action_drift'):.6f} accepted={m.get('accepted')}")
assert m.get("best_step") == steps, f"budget not honoured: {m.get('best_step')} != {steps}"
assert m.get("validations_run") == 1, "selection was not eliminated"
print("R1_BUDGET_VERIFIED")
PYV
