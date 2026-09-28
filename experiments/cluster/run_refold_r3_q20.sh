#!/usr/bin/env bash
#$ -N MotRefR3
#$ -cwd
#$ -q gpu@HOST_B
#$ -l gpu_card=1
#$ -pe smp 8
#$ -l h_rt=00:20:00
#$ -j y

set -euo pipefail
export HF_HOME=${WORK_ROOT}/hf_cache
export JAX_PLATFORMS=cpu
ROOT=${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
ADAPTER=$LR/round_curve/adapter/R3c_q20
OUT=$LR/motivation_search/checkpoints/R3c_q20
PY=$ROOT/external/VLA-Arena/envs/openpi/.venv/bin/python
cd "$ROOT"
[ -d "$ADAPTER/offset_0/params" ] || { echo "ABORT: missing adapter $ADAPTER"; exit 1; }
if [ -d "$OUT/offset_0/params" ]; then echo "SKIP: $OUT/offset_0 exists"; exit 0; fi
mkdir -p "$OUT"
echo "REFOLD_R3_START host=$(hostname -s) adapter=$ADAPTER output=$OUT"
/usr/bin/time -v "$PY" - "$ADAPTER" "$OUT" <<'PYFOLD'
import pathlib, sys
op = pathlib.Path("${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/models/openpi")
sys.path.insert(0, str(op / "scripts"))
import fold_lora_adapter as fold
import orbax.checkpoint as ocp
fold.FOLDS = pathlib.Path(sys.argv[1])
fold.OUT = pathlib.Path(sys.argv[2])
fold.OUT.mkdir(parents=True, exist_ok=True)
fold.fold_one(0, ocp.StandardCheckpointer(), verbose=True)
PYFOLD
[ -d "$OUT/offset_0/params" ] || { echo "ABORT: no folded params"; exit 1; }
du -sh "$OUT/offset_0"
echo "REFOLD_R3_DONE host=$(hostname -s) checkpoint=$OUT/offset_0"
