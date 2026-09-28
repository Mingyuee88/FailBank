#!/usr/bin/env bash
#$ -N SE_train
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 4
#$ -t 1-2
#$ -tc 2
#$ -o ${WORK_ROOT}/lora_dagger/se_r1/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/se_r1/batch/j.$JOB_ID.$TASK_ID.out
#
# Self-evolution round training. ROUND and SRC are set by the submitter via -v.
#
# Two arms over ONE record set (3657 records, identical in both), differing only in which
# records carry a real delta:
#   CURR    delta on S1_early only (225)            -- staged
#   STATIC  delta on S1_early+S2_mid+NO_RISK (1528) -- same data, unstaged
#
# The R1 defect this build fixes: R1 zeroed non-selected records, which on a failure bank
# means "target = the action that caused the crash". Its own fold carried 389 such targets
# against 225 real corrections, 1.7:1, and all four R1 arms lost 6-10 successes and roughly
# doubled crash count. Records whose target would be a bad action are now DROPPED (2144 of
# them: POST_crossing 1736, S3_emergency 225, failure-trajectory records with no real
# correction 183), and the builder asserts zero survive.
#
# Fixed budget, per PREREG_2026-08-16 S1/S3: exactly N steps, one validation, selection
# eliminated rather than equalised. 100 steps because every R1 arm cleared the registered
# quiet-drift guard there (flow ratio 0.80-0.89 against the 1.10 limit).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OP=external/VLA-Arena/vla_arena/models/openpi
ROUND=${ROUND:-1}
SEC=$LR/se_r${ROUND}
STEPS=${STEPS:-100}

ARMS=(CURR STATIC)
ROOTS=(curriculum_records static_records)
EXPECT=(225 1528)
i=$((SGE_TASK_ID - 1))
ARM=${ARMS[$i]}; ROOT=$SEC/${ROOTS[$i]}; WANT=${EXPECT[$i]}
OUT=$SEC/train/${ARM}
mkdir -p "$OUT" "$SEC/batch"
if [ -s "$OUT/offset_0/metrics.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"

external/VLA-Arena/envs/openpi/.venv/bin/python - "$ROOT" "$ARM" "$WANT" <<'PYCHK'
import json, math, sys, pathlib, collections
root, arm, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
def c3(a): return [max(-1.0, min(1.0, float(x))) for x in a[:3]]
n_delta = 0; stages = collections.Counter(); bad = 0; total = 0
for line in open(pathlib.Path(root) / "derived/folds/offset_0/train.jsonl"):
    r = json.loads(line); total += 1
    st = r.get("curriculum_stage")
    assert st not in ("POST_crossing", "S3_emergency"), f"dropped stage survived: {st}"
    d = math.dist(c3(r["nominal_action"]), r["executed_action"][:3])
    if d > 1e-9:
        n_delta += 1; stages[st] += 1
    elif not r.get("eventual_success"):
        bad += 1
print(f"ARM {arm}: records={total} real_delta={n_delta} stages={dict(stages)}")
assert n_delta == want, f"{arm} expected {want} deltas, found {n_delta}"
assert bad == 0, f"{bad} failure-trajectory records still target their own action"
assert total == 3657, f"record set changed: {total} != 3657 (arms must be dose-matched)"
print("SE_ROOT_VERIFIED")
PYCHK

echo "=== SE round=$ROUND arm=$ARM steps=$STEPS root=$ROOT on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  $OP/scripts/train_phase2_lora.py \
  --offset 0 --records-root "$ROOT/blobs" --derived-root "$ROOT/derived/folds" \
  --batch-size 2 --quiet-weight 0.0 \
  --smoke-steps "$STEPS" --validation-interval "$STEPS" --patience 99 \
  --output-root "$OUT"
echo "SE_TRAIN_EXIT=$? round=$ROUND arm=$ARM"

external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$STEPS" "$ARM" <<'PYV'
import json, sys, pathlib
out, steps, arm = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = json.load(open(pathlib.Path(out) / "offset_0/metrics.json"))
print(f"{arm}: best_step={m.get('best_step')} validations_run={m.get('validations_run')} "
      f"flow_ratio={m.get('best_quiet_flow_ratio'):.5f} "
      f"act_drift={m.get('best_quiet_action_drift'):.6f} accepted={m.get('accepted')}")
assert m.get("best_step") == steps and m.get("validations_run") == 1
print("SE_BUDGET_VERIFIED")
PYV
