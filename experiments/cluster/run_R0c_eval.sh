#!/usr/bin/env bash
#$ -N R0c_eval
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 4
#$ -t 1-4
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/sec_r0/batch_eval/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/sec_r0/batch_eval/j.$JOB_ID.$TASK_ID.out
#
# SEC R0c -- the primary endpoint for R0.
#
# R0 asks: does support-gating the training targets make support EXPANSION GENERALISE,
# where training on everything (arm D) did not? E12/E13 established that D's support moved
# toward the corrections it was trained on in-sample (high bucket 20 release / 0 regression)
# but not on held-out records (2/2, no movement; med+high pooled p = 0.289).
#
# So the endpoint is stage migration on the HELD-OUT records, not Arena success. Per
# PREREG_2026-08-16 rule S4, checkpoint assessment does not look at rollouts. It is also
# the only endpoint with usable power: dev is 47 cells with D at 41/47, where exact McNemar
# needs all six remaining failures fixed and none broken to reach p = 0.031.
#
# Each arm is folded into a full checkpoint and then annotated on the same 286 held-out
# records (offset 0) that E13 used for base and D, with the same K=8 and the same formula,
# so all six policies -- base, D, GATED, MATCHED, SHAM, ANTI -- sit on one scale.
#
# Budget: all four arms at 50 steps. The 150-step budget is NOT used: GATED, MATCHED and
# SHAM all failed the preregistered quiet-drift guard there while ANTI passed, which is
# reported as a result rather than worked around by relaxing a registered guard.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
SEC=$LR/sec_r0
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python
RECORDS=$LR/phase1_collection/records
STEPS=50

ARMS=(GATED MATCHED SHAM ANTI)
ARM=${ARMS[$((SGE_TASK_ID - 1))]}
SRC=$SEC/train_fixed/${ARM}_s${STEPS}
[ -s "$SRC/offset_0/metrics.json" ] || { echo "R0C_ABORT: no trained arm at $SRC"; exit 1; }

FOLD=$SEC/fold_fixed/${ARM}_s${STEPS}
OUT=$SEC/eval_fixed/${ARM}_s${STEPS}
mkdir -p "$FOLD" "$OUT" "$SEC/batch_eval"
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

# ---- fold the LoRA adapter into a full checkpoint ----
if [ ! -d "$FOLD/offset_0/params" ]; then
PYTHONPATH=src:external/VLA-Arena:. $VENV - "$SRC" "$FOLD" <<'PYFOLD'
import pathlib, sys
OP = pathlib.Path("${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/models/openpi")
sys.path.insert(0, str(OP / "scripts"))
import fold_lora_adapter as F
import orbax.checkpoint as ocp
F.FOLDS = pathlib.Path(sys.argv[1])
F.OUT = pathlib.Path(sys.argv[2])
F.OUT.mkdir(parents=True, exist_ok=True)
print(f"FOLDS={F.FOLDS}  OUT={F.OUT}  BASE={F.BASE}")
F.fold_one(0, ocp.StandardCheckpointer(), verbose=True)
print("FOLD_OK")
PYFOLD
fi
CK=$FOLD/offset_0
[ -d "$CK/params" ] || { echo "R0C_ABORT: fold produced no params at $CK"; exit 1; }
# the served checkpoint needs the assets dir alongside params, as v20_fold arms have
[ -e "$CK/assets" ] || ln -s ${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned/assets "$CK/assets"

# ---- annotate the held-out records under this policy ----
PORT=$((53000 + SGE_TASK_ID))
SERVE=external/VLA-Arena/vla_arena/models/openpi/scripts/serve_policy.py
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

echo "=== R0c arm=$ARM ckpt=$CK port=$PORT on $(hostname) ==="
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
  --records-root "$RECORDS" --fold offset_0 --split validation.jsonl \
  --out "$OUT/support.jsonl" --k 8 --port "$PORT" 2>&1 | tee "$OUT/annotate.log"

grep -q "SELF-CHECK PASSED" "$OUT/annotate.log" || {
  echo "R0C_FAILED arm=$ARM -- self-check did not pass"; rm -f "$OUT/support.jsonl"; exit 1; }
echo "SELF-CHECK_PASSED" > "$OUT/done.txt"
wc -l "$OUT/support.jsonl"
echo "R0C_ARM_VERIFIED arm=$ARM"
