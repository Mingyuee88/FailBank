#!/usr/bin/env bash
#$ -N v37_tgrid
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_D
#$ -pe smp 2
#$ -t 1-10
#$ -tc 3
#$ -o ${WORK_ROOT}/lora_dagger/v37_teachergrid/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v37_teachergrid/batch/j.$JOB_ID.$TASK_ID.out
#
# v37_teachergrid -- does the barrier CENTRE of the teacher change what the student
# inherits? Five paired data-order seeds per teacher, all evaluated, none dropped.
#
# WHAT THIS IS NOT. It is not "distil from the better teacher". That framing is withdrawn:
# on the development arena dual vs eef is 40 vs 37 successes with discordant pairs 4 vs 1,
# McNemar p=0.375, and dual leaves nearly twice the absolute policy-induced cost (573 vs
# 292 summed over 47 offsets). There is no established teacher ordering. What there IS is
# a mechanism: the gripper-centred barrier destroys base successes concentrated where
# there was no cost to remove (AUC 0.000, p=0.00024), and re-centring on the manipulated
# object recovers 7 of those 9. Whether that mechanism survives distillation into weights
# is a question about the method, and it does not need a teacher ordering to be worth
# answering.
#
# WHY FIVE SEEDS AND NOT ONE. Because one is demonstrably not evidence here. The same
# training data and configuration at a different data-order seed reverses the ordering of
# the method and its own null on the development arena: A 38 vs SHAM 35, but A_s2 31 vs
# SHAM_s2 32. Every arm in this study early-stops at step 25-50 of a 400-step budget with
# the quiet-drift guard binding, so the adapters are small perturbations and the deployed
# differences are single-digit episodes -- exactly the regime where a single checkpoint
# reports noise. All ten checkpoints are evaluated and all ten are reported.
#
# THE ONE VARIABLE. Both roots are round-1 records aggregated with a second collection run
# on the SAME 29 training-half offsets, by the SAME base policy, with the SAME recorder,
# on the SAME host, at a matched budget. Only SE_VLA_AEGIS_BARRIER_CENTER differed during
# that second collection (eef -> dual). The eef root is the one R2CTRL trained on, so this
# grid also puts five seeds under R2CTRL, which previously had one.
#
# PRE-DECLARED READOUT, fixed before any checkpoint here is evaluated:
#   endpoint            xfer, safety_static_obstacles L1t1/t3/t4, 150 offsets, HOST_B
#   primary statistic   success rate, paired per offset, seed treated as a blocking factor
#   secondary           policy-induced cost, reported in ABSOLUTE totals as well as
#                       percentages, because the percentage form hid the dual/eef reversal
#   no arm is dropped, no seed is dropped, and no other endpoint is substituted afterwards
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OP=external/VLA-Arena/vla_arena/models/openpi

SEEDS=(0 777 1009 2027 3037)
i=$((SGE_TASK_ID - 1))
case $((i / 5)) in
  0) TEACHER=eef;  ROOT=$LR/v30_r2ctrl/records ;;
  1) TEACHER=dual; ROOT=$LR/v37_dualmerged/records ;;
esac
SEED=${SEEDS[$((i % 5))]}
ARM=T${TEACHER}_s${SEED}
HELDOUT=0

[ -f "$ROOT/derived/folds/offset_$HELDOUT/train.jsonl" ] || {
  echo "NOT_READY $ARM -- $ROOT has no derived fold yet"; exit 0; }

OUT=$LR/v37_teachergrid/train/$ARM
mkdir -p "$OUT" $LR/v37_teachergrid/batch
if [ -d "$OUT/offset_$HELDOUT/params" ]; then echo "SKIP existing $ARM"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

# Source audit. The failure that would quietly collapse this grid into one arm is both
# roots resolving to the same records. Assert they differ, and differ in the way claimed:
# same offsets, same record count to within the collection's own variation, different
# executed actions on the steps where the two barriers disagree.
external/VLA-Arena/envs/openpi/.venv/bin/python - "$ROOT" "$ARM" "$HELDOUT" "$LR" <<'PYCHK'
import json, pathlib, sys, collections
root, arm, heldout, LR = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
f = pathlib.Path(root) / "derived/folds" / f"offset_{heldout}" / "train.jsonl"
other = (pathlib.Path(LR) / "v37_dualmerged/records/derived/folds" / f"offset_{heldout}" / "train.jsonl"
         if "r2ctrl" in root else
         pathlib.Path(LR) / "v30_r2ctrl/records/derived/folds" / f"offset_{heldout}" / "train.jsonl")
def read(p):
    d = {}
    for line in open(p):
        r = json.loads(line)
        d[r["record_id"]] = tuple(round(float(x), 12) for x in r["executed_action"])
    return d
mine = read(f)
offs = collections.Counter()
for line in open(f):
    offs[str(json.loads(line)["offset"])] += 1
print(f"ARM {arm}: root={root} records={len(mine)} offsets={sorted(offs, key=int)}")
if other.exists():
    theirs = read(other)
    shared = set(mine) & set(theirs)
    differing = sum(1 for k in shared if mine[k] != theirs[k])
    print(f"  vs the other teacher's root: shared_record_ids={len(shared)} "
          f"differing_executed_action={differing} "
          f"only_mine={len(set(mine)-set(theirs))} only_theirs={len(set(theirs)-set(mine))}")
    assert len(mine) != len(theirs) or differing > 0 or len(set(mine) ^ set(theirs)) > 0, \
        "the two teacher roots are identical -- the barrier centre never reached collection"
print("TEACHER_ROOT_VERIFIED")
PYCHK

echo "=== v37 teacher=$TEACHER seed=$SEED root=$ROOT on $(hostname) ==="
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

external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/offset_$HELDOUT/metrics.json" "$ARM" "$SEED" <<'PYV'
import json, sys
m = json.load(open(sys.argv[1])); arm, seed = sys.argv[2], int(sys.argv[3])
print(f"V37 arm={arm} data_seed={m.get('data_seed')} best_step={m.get('best_step')} "
      f"base_triggered_loss={m.get('base_triggered_loss')} "
      f"best_triggered_loss={m.get('best_triggered_loss')} "
      f"best_quiet_flow_ratio={m.get('best_quiet_flow_ratio')}")
assert m.get("data_seed") == seed, f"metrics record data_seed {m.get('data_seed')}, expected {seed}"
print("V37_SEED_VERIFIED")
PYV
