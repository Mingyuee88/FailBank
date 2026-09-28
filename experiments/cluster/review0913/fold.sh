#!/usr/bin/env bash
#$ -N RvFold
#$ -cwd
#$ -l gpu_card=1
#$ -pe smp 8
#$ -j y
# Review 2026-09-13. Fold one LoRA adapter onto a base checkpoint.
# LIST line: ADAPTER_DIR OUT_DIR BASE_CKPT
# Refolding was verified bit-identical to the original fold (job 1421679, 51/51 leaves,
# max abs diff 0). Peak RSS 52.6 GB, so this must run on a large node, never the frontend.
set -euo pipefail
: "${LIST:?set LIST}"
cd ${SE_VLA_ROOT}
export JAX_PLATFORMS=cpu HF_HOME=${WORK_ROOT}/hf_cache
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python
read -r ADAPTER OUT BASE <<< "$(sed -n "${SGE_TASK_ID}p" "$LIST")"
echo "=== RvFold task=$SGE_TASK_ID adapter=$ADAPTER out=$OUT base=$BASE host=$(hostname) ==="
[ -d "$ADAPTER/offset_0/params" ] || { echo "FOLD_ABORT no adapter params at $ADAPTER"; exit 1; }
[ -d "$BASE/params" ] || { echo "FOLD_ABORT no base params at $BASE"; exit 1; }
if [ -d "$OUT/offset_0/params" ]; then echo "FOLD_SKIP $OUT already folded"; exit 0; fi
mkdir -p "$OUT"
SE_VLA_FOLD_BASE="$BASE" PYTHONPATH=src:external/VLA-Arena:. $VENV - "$ADAPTER" "$OUT" <<"PY"
import pathlib, sys, os
OP = pathlib.Path("${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/models/openpi")
sys.path.insert(0, str(OP / "scripts"))
import fold_lora_adapter as F
import orbax.checkpoint as ocp
F.FOLDS = pathlib.Path(sys.argv[1]); F.OUT = pathlib.Path(sys.argv[2])
F.BASE = pathlib.Path(os.environ["SE_VLA_FOLD_BASE"]).resolve()
F.OUT.mkdir(parents=True, exist_ok=True)
n = F.fold_one(0, ocp.StandardCheckpointer(), verbose=False)
print("FOLD_OK folded=%d base=%s" % (n, F.BASE))
PY
[ -d "$OUT/offset_0/params" ] || { echo "FOLD_FAILED $OUT"; exit 1; }
case "$BASE" in
  *pi0_vla_arena_finetuned*) $VENV roundG/pi05_stage2/lora/verify_family.py "$OUT/offset_0/params" pi0 ;;
esac
echo "FOLD_DONE $OUT"
