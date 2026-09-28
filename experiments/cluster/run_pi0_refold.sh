#!/bin/bash
#$ -N Pi0Fold
#$ -cwd
#$ -q gpu@HOST_A
#$ -l gpu_card=1
#$ -pe smp 4
#$ -j y
# Re-fold the existing pi0 LoRA adapter onto the CORRECT (pi0) base.
# The original ckpt was folded onto fold_lora_adapter.py's hardcoded pi05 BASE, which
# produced a pi05-structured checkpoint from a pi0 training run. The adapter itself
# (20 LoRA leaves) is base-agnostic, so no retraining is needed -- only a re-merge.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
cd ${SE_VLA_ROOT}
L=${WORK_ROOT}/lora_dagger
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python
BASE=${SE_VLA_ROOT}/checkpoints/pi0_vla_arena_finetuned
# Write to a fresh path each attempt: orbax commits via rename into the final name and
# a half-written previous attempt on NFS makes that rename fail with a missing tmp dir.
OUT=$L/se_pi0_q05/ckpt_pi0fold_${JOB_ID:-manual}/PI0_Q05
rm -rf "$OUT" 2>/dev/null || true
mkdir -p "$OUT"
echo "SPACE_BEFORE: $(df -BG ${GROUP_ROOT} | tail -1 | awk "{print \$4}") avail"
echo "OUT=$OUT"
SE_VLA_FOLD_BASE=$BASE PYTHONPATH=src:external/VLA-Arena:. $VENV - \
  "$L/se_pi0_q05/adapter/PI0_Q05" "$OUT" <<'PYFOLD'
import pathlib, sys, os
OP = pathlib.Path("${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/models/openpi")
sys.path.insert(0, str(OP / "scripts"))
import fold_lora_adapter as F
import orbax.checkpoint as ocp
F.FOLDS = pathlib.Path(sys.argv[1]); F.OUT = pathlib.Path(sys.argv[2])
F.BASE = pathlib.Path(os.environ["SE_VLA_FOLD_BASE"]).resolve()
print("FOLD_BASE=%s" % F.BASE)
F.OUT.mkdir(parents=True, exist_ok=True)
F.fold_one(0, ocp.StandardCheckpointer(), verbose=True)
print("FOLD_OK")
PYFOLD
echo "=== verify ==="
$VENV roundG/pi05_stage2/lora/verify_family.py "$OUT/offset_0/params" pi0
echo "REFOLD_DONE"
