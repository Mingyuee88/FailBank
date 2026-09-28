#!/usr/bin/env bash
#$ -N v19_train
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_D
#$ -pe smp 2
#$ -t 1-4
#$ -tc 2
#$ -o ${WORK_ROOT}/lora_dagger/v19_train/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v19_train/batch/j.$JOB_ID.$TASK_ID.out
#
# v19_train -- the four preregistered LoRA arms.
#
# WHAT IS BEING TESTED. The LoRA distillation line previously reached fix rate 40%
# with retention 88.2%, i.e. it repaired failures and destroyed 11.8% of the
# successes it should have left alone, netting about zero. The cause is in the
# collection, not the training:
#
#     phase1_collect.py OFFSETS = the 10 base-FAILURE offsets, and nothing else
#     old fold offset_0: 2830 records, quiet-from-SUCCESS-episodes = 0 (verified)
#
# Every teacher record came from an episode the base policy was going to fail, so
# the model never saw the state distribution of a successful grasp paired with a
# "do nothing" target, and learned to correct everywhere. The `quiet_weight` knob
# could not fix that and its four-arm sweep correctly found nothing: it reweights
# quiet STEPS INSIDE FAILING episodes, a different distribution.
#
#     new fold offset_0: 6305 records, quiet-from-SUCCESS-episodes = 2133 (verified)
#
# ARMS. All four are required. Arm C alone proving out would mean nothing -- A and
# B exist to show the old result reproduces and that quiet_weight alone still does
# not, so any gain in C is attributable to the negative data rather than to the
# rebuild, the reweighting, or run-to-run variation.
#
#   A  old records, quiet_weight 0.0   reproduce the 88.2% retention baseline
#   B  old records, quiet_weight 0.3   reproduce the failure of the quiet sweep
#   C  new records, quiet_weight 0.3   the change under test
#   D  new records, quiet_weight 0.0   isolates negative DATA from quiet REWEIGHTING
#
# LEAKAGE. The merge put base-success offsets into training, which would have made
# a within-task retention number circular -- retention is measured on exactly those
# offsets. v18_split therefore uses only every other success offset for training and
# keeps the rest unseen. The split was fixed before any training and is recorded in
# v18_split/split.json.
#
# ★ THE CLAIM IS NOT ALLOWED TO REST ON L1t2. Final judgement is the per-task
# transfer to L1t1/L1t3/L1t4, where nothing has been seen. geo5 scored 4/6 on this
# development domain and then destroyed half the successes on transfer; a good
# within-task number here proves nothing by itself.
#
# All four train on ONE pinned host: GPU model changes numerics, and a four-way
# comparison must not carry that term.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export WANDB_MODE=disabled
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OP=external/VLA-Arena/vla_arena/models/openpi

OLD=$LR/phase1_collection/records
NEW=$LR/v18_split/records
case "$SGE_TASK_ID" in
  1) ARM=A_old_qw0.0;  ROOT=$OLD; QW=0.0 ;;
  2) ARM=B_old_qw0.3;  ROOT=$OLD; QW=0.3 ;;
  3) ARM=C_new_qw0.3;  ROOT=$NEW; QW=0.3 ;;
  4) ARM=D_new_qw0.0;  ROOT=$NEW; QW=0.0 ;;
esac
# held-out validation fold: a base-FAILURE offset present in both roots, so all
# four arms validate on the same thing
HELDOUT=0

OUT=$LR/v19_train/$ARM
mkdir -p "$OUT" $LR/v19_train/batch
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

# source audit before spending an hour of GPU: confirm this arm's records root
# really is the one intended, by counting quiet records drawn from success episodes
external/VLA-Arena/envs/openpi/.venv/bin/python - "$ROOT" "$ARM" <<'PYCHK'
import json, sys, glob, os
root, arm = sys.argv[1], sys.argv[2]
FAIL = {"0","5","10","26","27","29","35","36","38","42","43","47"}
f = sorted(glob.glob(f"{root}/derived/folds/*/train.jsonl"))[0]
qs = qf = n = 0
for line in open(f):
    r = json.loads(line); n += 1
    if r.get("quiet"):
        if str(r["offset"]) in FAIL: qf += 1
        else: qs += 1
print(f"ARM {arm}: records={n} quiet_from_failure={qf} quiet_from_success={qs}")
if arm.startswith(("C_", "D_")):
    assert qs > 500, f"NEW root must carry success-episode quiet records, got {qs}"
else:
    assert qs == 0, f"OLD root must carry none, got {qs}"
print("RECORDS_ROOT_VERIFIED")
PYCHK

echo "=== v19 arm=$ARM root=$ROOT quiet_weight=$QW heldout=$HELDOUT on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  $OP/scripts/train_phase2_lora.py \
  --offset "$HELDOUT" \
  --records-root "$ROOT/blobs" \
  --derived-root "$ROOT/derived/folds" \
  --batch-size 2 \
  --quiet-weight "$QW" \
  --output-root "$OUT"
echo "TRAIN_EXIT=$? arm=$ARM"
