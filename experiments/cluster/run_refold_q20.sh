#!/usr/bin/env bash
#$ -N MotRefold
#$ -cwd
#$ -q gpu@HOST_A
#$ -l gpu_card=1
#$ -pe smp 8
#$ -l h_rt=00:20:00
#$ -j y

# Reconstruct the inference checkpoint from the archived, lossless LoRA adapter.
# Folding is CPU/numpy; the GPU request places the job on a node with enough RAM and
# pins it to the host used by the matched motivation evaluations.
set -euo pipefail
export HF_HOME=${WORK_ROOT}/hf_cache
export JAX_PLATFORMS=cpu

ROOT=${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
ADAPTER=$LR/nocurr/adapter/R2_q20
OUT=$LR/motivation_search/checkpoints/R2_q20
PY=$ROOT/external/VLA-Arena/envs/openpi/.venv/bin/python

cd "$ROOT"
[ -d "$ADAPTER/offset_0/params" ] || { echo "ABORT: missing adapter $ADAPTER"; exit 1; }
if [ -d "$OUT/offset_0/params" ]; then
  echo "SKIP: folded checkpoint already exists at $OUT/offset_0"
  exit 0
fi
mkdir -p "$OUT"
echo "REFOLD_START host=$(hostname -s) adapter=$ADAPTER output=$OUT"

/usr/bin/time -v "$PY" - "$ADAPTER" "$OUT" <<'PYFOLD'
import pathlib
import sys

op = pathlib.Path("${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/models/openpi")
sys.path.insert(0, str(op / "scripts"))
import fold_lora_adapter as fold
import orbax.checkpoint as ocp

fold.FOLDS = pathlib.Path(sys.argv[1])
fold.OUT = pathlib.Path(sys.argv[2])
fold.OUT.mkdir(parents=True, exist_ok=True)
fold.fold_one(0, ocp.StandardCheckpointer(), verbose=True)
PYFOLD

[ -d "$OUT/offset_0/params" ] || { echo "ABORT: folding produced no params"; exit 1; }
du -sh "$OUT/offset_0"
echo "REFOLD_DONE host=$(hostname -s) checkpoint=$OUT/offset_0"
