#!/usr/bin/env bash
#$ -N p2_repev
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-45
#$ -tc 3
#$ -o ${WORK_ROOT}/lora_dagger/phase2_repeat_eval/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/phase2_repeat_eval/batch/j.$JOB_ID.$TASK_ID.out
#
# Deploy-and-evaluate each training repeat on its own held-out offset, so the
# distribution of the DEPLOYED outcome over training runs can be read off.
# Pairs with run_repeat.sh (job holds until that array finishes).
#
# A repeat whose guard rejected every validation writes no adapter. That is not an
# error -- it is a draw of the deployment pipeline that yields nothing deployable,
# and it is recorded as such (no_deployable_checkpoint) rather than skipped.
#
# Fold -> evaluate -> DELETE: a merged checkpoint is 12 GB, so peak extra usage is
# one per concurrent task (3), not 45.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers PI05_SERVER_TIMEOUT_SEC=600
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
cd ${SE_VLA_ROOT}

OFFSETS=(0 5 10 26 27 29 38 42 43)
i=$((SGE_TASK_ID-1))
OFF=${OFFSETS[$((i%9))]}
REP=$((i/9+1))

SRC=${WORK_ROOT}/lora_dagger/phase2_repeat/rep${REP}
OUT=${WORK_ROOT}/lora_dagger/phase2_repeat_eval/rep${REP}_off${OFF}
TMP=${WORK_ROOT}/lora_dagger/phase2_repeat_merged/r${REP}_o${OFF}
mkdir -p "$OUT" ${WORK_ROOT}/lora_dagger/phase2_repeat_eval/batch
cleanup() { rm -rf "$TMP" 2>/dev/null || true; }
trap cleanup EXIT

if [ ! -d "$SRC/offset_${OFF}/params" ]; then
  echo "NO_DEPLOYABLE_CHECKPOINT rep=$REP offset=$OFF"
  printf '{"rep":%d,"offset":%d,"status":"no_deployable_checkpoint"}\n' "$REP" "$OFF" > "$OUT/result.json"
  exit 0
fi

mkdir -p "$TMP"
echo "=== rep $REP fold $OFF: folding ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python - "$OFF" "$SRC" "$TMP" <<'PY'
import sys, pathlib, os, shutil
OP = pathlib.Path("${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/models/openpi")
sys.path.insert(0, str(OP / "scripts"))
import fold_lora_adapter as F
import orbax.checkpoint as ocp
off, src, tmp = int(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
# the repeat trainer already writes offset_<N>/params, which is exactly the layout
# fold_one expects -- no symlink shim needed here.
F.FOLDS = src
F.OUT = tmp.parent
os.makedirs(tmp.parent, exist_ok=True)
n = F.fold_one(off, ocp.StandardCheckpointer())
produced = tmp.parent / f"offset_{off}"
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

echo "=== rep $REP fold $OFF: evaluating ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm base --port-base 48000 \
  --array-task-id "$SGE_TASK_ID" --server-attempts 3
echo "REPEV_EXIT=$? rep=$REP offset=$OFF"
