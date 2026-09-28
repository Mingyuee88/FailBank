#!/usr/bin/env bash
#$ -N denseev
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-32
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/phase2_dense_eval/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/phase2_dense_eval/batch/j.$JOB_ID.$TASK_ID.out
# Fold -> evaluate -> DELETE, one checkpoint per task.
# A merged checkpoint is 12 GB and the earlier sweep left 384 GB behind on a
# filesystem already at 95%. Here the merged copy exists only for the duration of
# its own evaluation, so peak extra usage is one checkpoint rather than 32.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers PI05_SERVER_TIMEOUT_SEC=600
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
cd ${SE_VLA_ROOT}

LIST=${WORK_ROOT}/lora_dagger/phase2_dense_eval/jobs.txt
LINE=$(sed -n "${SGE_TASK_ID}p" "$LIST")
[ -n "$LINE" ] || { echo "no job for task $SGE_TASK_ID"; exit 0; }
FOLD=$(echo "$LINE" | awk '{print $1}')
STEP=$(echo "$LINE" | awk '{print $2}')

DENSE=${WORK_ROOT}/lora_dagger/phase2_dense
OUT=${WORK_ROOT}/lora_dagger/phase2_dense_eval/fold${FOLD}_step${STEP}
TMP=${WORK_ROOT}/lora_dagger/phase2_dense_merged/f${FOLD}_s${STEP}
mkdir -p "$OUT" "$TMP"
cleanup() { rm -rf "$TMP" 2>/dev/null || true; }
trap cleanup EXIT

echo "=== fold $FOLD step $STEP: folding ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python - "$FOLD" "$STEP" "$TMP" <<'PY'
import sys, pathlib, os, shutil
OP = pathlib.Path("${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/models/openpi")
sys.path.insert(0, str(OP / "scripts"))
import fold_lora_adapter as F
import orbax.checkpoint as ocp
fold, step, tmp = int(sys.argv[1]), int(sys.argv[2]), pathlib.Path(sys.argv[3])
src = pathlib.Path("${WORK_ROOT}/lora_dagger/phase2_dense") / f"offset_{fold}"
link = src / f"offset_{step}"
if not link.exists():
    os.symlink(src / f"step_{step:04d}", link)
F.FOLDS = src
F.OUT = tmp.parent
os.makedirs(tmp.parent, exist_ok=True)
n = F.fold_one(step, ocp.StandardCheckpointer())
produced = tmp.parent / f"offset_{step}"
if produced != tmp and produced.exists():
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.move(str(produced), str(tmp))
assets = tmp / "assets"
if not assets.exists():
    os.symlink(pathlib.Path(F.BASE) / "assets", assets)
print(f"folded {n} weights -> {tmp}")
PY

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_SRD_WRITES_ENABLED=0
unset SE_VLA_PHASE1_RECORD || true
export SE_VLA_POLICY_CHECKPOINT_DIR="$TMP"
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0

echo "=== fold $FOLD step $STEP: evaluating ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$FOLD" --seed 23 \
  --replan-steps 1 --trials 1 --arm base --port-base $((46000 + SGE_TASK_ID)) \
  --array-task-id "$SGE_TASK_ID" --server-attempts 3
echo "DENSEEVAL_EXIT=$? fold=$FOLD step=$STEP"
