#!/usr/bin/env bash
#$ -N E12_annot
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 4
#$ -t 1-2
#$ -tc 2
#$ -o ${WORK_ROOT}/E12_annot/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/E12_annot/batch/j.$JOB_ID.$TASK_ID.out
#
# E12 -- offline action-support annotation of the phase-1 learning records.
#
# Every record gets the quantity the curriculum stages on: how far the teacher's corrected
# action lies from the policy's own action distribution at that observation. Task 1 scores
# the records under BASE, task 2 under D. The two together answer the question E11 could
# not: D's training set contained records at every support distance, so does the subset
# that was INSIDE base's support behave differently from the subset that was outside?
#
# This is not a rollout. No simulator, no shield, no episode. It re-asks the policy the
# archived question and samples K=8 chunks, using the formula copied verbatim from
# `_vlsa_dispersion` in run.py so the numbers are comparable with E10/E11.
#
# Why it cannot be done inside collection: run.py:93 refuses SE_VLA_VLSA_DISPERSION together
# with SE_VLA_PHASE1_RECORD (the InferCapture pops by call index and extra sampling calls
# desynchronise it), and the probe perturbs the sampler anyway. Offline is also the only
# form that can be re-run against a new policy each round, which the curriculum requires.
#
# The load-bearing check is SELF-CHECK: the policy's own nominal action must land inside its
# own sample cloud (E10 in-rollout reference: median d_nominal_min 0.0733). If the
# reconstructed observation were not the one the policy actually saw, that number would blow
# up and every support annotation would be noise. The run fails closed on it.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}

RECORDS=${WORK_ROOT}/lora_dagger/phase1_collection/records
OUTROOT=${WORK_ROOT}/E12_annot
FOLD=offset_0
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python

case "$SGE_TASK_ID" in
  1) ARM=base; CK=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned ;;
  2) ARM=D;    CK=${WORK_ROOT}/lora_dagger/v20_fold/D_new_qw0.0/offset_0 ;;
  *) echo "no arm for task $SGE_TASK_ID"; exit 0 ;;
esac
[ -d "$CK" ] || { echo "E12_ABORT: missing checkpoint $CK"; exit 1; }

OUT=$OUTROOT/$ARM
mkdir -p "$OUT" "$OUTROOT/batch"
if [ -s "$OUT/support.jsonl" ] && grep -q SELF-CHECK_PASSED "$OUT/done.txt" 2>/dev/null; then
  echo "SKIP $OUT"; exit 0
fi
rm -f "$OUT/support.jsonl" "$OUT/done.txt"
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

PORT=$((51000 + SGE_TASK_ID))
SERVE=external/VLA-Arena/vla_arena/models/openpi/scripts/serve_policy.py
[ -f "$SERVE" ] || { echo "E12_ABORT: no serve_policy.py at $SERVE"; exit 1; }

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

echo "=== E12 arm=$ARM ckpt=$CK port=$PORT on $(hostname) ==="
setsid $VENV "$SERVE" --port "$PORT" policy:checkpoint \
  --policy.config pi05_vla_arena --policy.dir "$CK" > "$OUT/server.log" 2>&1 &
SPID=$!
cleanup() { kill -- -"$SPID" 2>/dev/null || kill "$SPID" 2>/dev/null || true; }
trap cleanup EXIT

for i in $(seq 1 180); do
  if $VENV -c "
import socket,sys
s=socket.socket(); s.settimeout(2)
sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" 2>/dev/null; then
    echo "server up after ${i}0s"; break
  fi
  sleep 10
done

$VENV roundG/pi05_stage2/lora/support_annotate.py \
  --records-root "$RECORDS" --fold "$FOLD" --split train.jsonl \
  --out "$OUT/support.jsonl" --k 8 --port "$PORT" 2>&1 | tee "$OUT/annotate.log"

grep -q "SELF-CHECK PASSED" "$OUT/annotate.log" || {
  echo "E12_FAILED arm=$ARM -- self-check did not pass"; rm -f "$OUT/support.jsonl"; exit 1; }
echo "SELF-CHECK_PASSED" > "$OUT/done.txt"
wc -l "$OUT/support.jsonl"
echo "E12_ARM_VERIFIED arm=$ARM"
