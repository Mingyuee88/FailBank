#!/usr/bin/env bash
#$ -N qw_eval
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-27
#$ -tc 12
#$ -o ${WORK_ROOT}/lora_dagger/phase2_qw_eval/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/phase2_qw_eval/batch/j.$JOB_ID.$TASK_ID.out
#
# STAGE A, shot 1 -- evaluation.
#
# One task per (quiet_weight, fold) adapter: fold it ONCE into a merged checkpoint,
# run all 16 evaluations against that one copy, then delete it. Peak extra disk is
# 12 GB x 12 concurrent = 144 GB rather than 27 x 12 GB = 324 GB, and each 12 GB
# checkpoint is written once instead of 16 times.
#
# Per adapter:
#   1 cell  -- its own held-out base-failure offset  -> FIX RATE
#  15 cells -- the frozen retention probe below      -> RETENTION
#
# PROBE FROZEN 2026-08-08 before any quiet_weight adapter was evaluated. Stratified
# by how often the 9 quiet_weight=0 adapters broke each offset in Phase3b, so the
# comparison is like-for-like: the qw=0 arm's retention is recomputed on these same
# 15 offsets rather than against its 37-offset average.
#   fragile      (broken by >=3 of 9):  1 12 17 37 40
#   intermediate (broken by 1-2 of 9):  2  8 13 19 22
#   never broken                     :  4 20 24 31 44
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
cd ${SE_VLA_ROOT}

OFFSETS=(0 5 10 26 27 29 38 42 43)
TAGS=(01 03 10)
i=$((SGE_TASK_ID-1))
FOLD=${OFFSETS[$((i%9))]}
TAG=${TAGS[$((i/9))]}
PROBE=(1 12 17 37 40 2 8 13 19 22 4 20 24 31 44)

LR=${WORK_ROOT}/lora_dagger
SRC=$LR/phase2_qw/qw${TAG}
# BUGFIX 2026-08-08: the first attempt set fold_one's output root to the SHARED
# phase2_qw_merged/, so it produced offset_<FOLD> -- a name that carries the fold
# but NOT the quiet_weight. The three arms of the same fold are 9 array tasks
# apart while -tc was 12, so they overlapped and wrote the same 12 GB directory:
# 5 tasks died on the OCDBT lock and the survivors may have picked up another
# arm's weights. Every task now folds inside its own task-numbered directory, so
# no two tasks can ever share a path regardless of -tc.
WORK=$LR/phase2_qw_merged/t${SGE_TASK_ID}
TMP=$WORK/offset_${FOLD}
mkdir -p "$LR/phase2_qw_eval/batch"
cleanup() { rm -rf "$WORK" 2>/dev/null || true; }
trap cleanup EXIT

if [ ! -d "$SRC/offset_${FOLD}/params" ]; then
  echo "NO_ADAPTER qw=$TAG fold=$FOLD (training produced no guard-passing checkpoint)"
  exit 0
fi

rm -rf "$WORK"; mkdir -p "$WORK"
echo "=== qw$TAG fold $FOLD: folding once into $WORK ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python - "$FOLD" "$SRC" "$WORK" <<'PY'
import sys, pathlib, os
OP = pathlib.Path("${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/models/openpi")
sys.path.insert(0, str(OP / "scripts"))
import fold_lora_adapter as F
import orbax.checkpoint as ocp
off, src, work = int(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
F.FOLDS = src
F.OUT = work            # task-private; fold_one writes work/offset_<off> directly
os.makedirs(work, exist_ok=True)
n = F.fold_one(off, ocp.StandardCheckpointer())
tmp = work / f"offset_{off}"
assets = tmp / "assets"
if not assets.exists():
    os.symlink(pathlib.Path(F.BASE) / "assets", assets)
print(f"folded {n} weights -> {tmp}")
PY

export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_SRD_WRITES_ENABLED=0
unset SE_VLA_PHASE1_RECORD || true
export SE_VLA_POLICY_CHECKPOINT_DIR="$TMP"
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0
export SE_VLA_SEED=23

k=0
for OFF in "$FOLD" "${PROBE[@]}"; do
  k=$((k+1))
  OUT=$LR/phase2_qw_eval/qw${TAG}_f${FOLD}/off${OFF}
  mkdir -p "$OUT"
  if [ -s "$OUT/result.json" ]; then echo "skip off$OFF (done)"; continue; fi
  export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
  export SE_VLA_RESULT_PATH="$OUT/result.json"
  echo "--- qw$TAG fold$FOLD -> offset $OFF ($k/16) ---"
  # one bad cell must not abandon the other 15 or leak the 12 GB merged copy
  PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
    roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
    --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
    --replan-steps 1 --trials 1 --arm base --port-base $((54000 + 40 * SGE_TASK_ID)) \
    --array-task-id "$k" --server-attempts 3 || echo "CELL_FAILED qw=$TAG fold=$FOLD off=$OFF"
done
echo "QWEVAL_DONE qw=$TAG fold=$FOLD"
