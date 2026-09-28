#!/usr/bin/env bash
#$ -N v20_fold
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(HOST_D*|HOST_E*)
#$ -pe smp 2
#$ -t 1-4
#$ -tc 2
#$ -o ${WORK_ROOT}/lora_dagger/v20_fold/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v20_fold/batch/j.$JOB_ID.$TASK_ID.out
#
# v20_fold -- fold each of the four trained LoRA adapters into a deployable
# checkpoint. fold_lora_adapter.fold_one() reads FOLDS/offset_<N>/params and writes
# OUT/offset_<N>, which is exactly the layout train_phase2_lora produced under
# v19_train/<arm>/offset_0/params, so the folding code is used unchanged and only
# its two module constants are repointed.
#
# Each merged checkpoint is ~12 GB. The output directory carries the ARM NAME:
# a past run lost work because fold outputs from different factors collided in one
# directory, and four arms differing only in training data would collide silently.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

ARMS=(A_old_qw0.0 B_old_qw0.3 C_new_qw0.3 D_new_qw0.0)
ARM=${ARMS[$((SGE_TASK_ID - 1))]}
mkdir -p $LR/v20_fold/batch $LR/v20_fold/$ARM

if [ -d "$LR/v20_fold/$ARM/offset_0" ] && [ -n "$(ls -A "$LR/v20_fold/$ARM/offset_0" 2>/dev/null)" ]; then
  echo "SKIP existing $ARM"; exit 0
fi

echo "=== v20 fold arm=$ARM on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python - "$ARM" <<'PYFOLD'
import pathlib, sys
OP = pathlib.Path("${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/models/openpi")
sys.path.insert(0, str(OP / "scripts"))
import fold_lora_adapter as F
import orbax.checkpoint as ocp

arm = sys.argv[1]
LR = pathlib.Path("${WORK_ROOT}/lora_dagger")
F.FOLDS = LR / "v19_train" / arm
F.OUT = LR / "v20_fold" / arm
F.OUT.mkdir(parents=True, exist_ok=True)
print(f"FOLDS={F.FOLDS}  OUT={F.OUT}  BASE={F.BASE}")
ck = ocp.StandardCheckpointer()
F.fold_one(0, ck, verbose=True)
print("FOLD_OK", arm)
PYFOLD
echo "FOLD_EXIT=$? arm=$ARM"
du -sh $LR/v20_fold/$ARM 2>/dev/null || true
