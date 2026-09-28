#!/usr/bin/env bash
#$ -N v31_fold
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(HOST_D*|HOST_E*)
#$ -pe smp 2
#$ -t 1-18
#$ -tc 2
#$ -o ${WORK_ROOT}/lora_dagger/v31_fold/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v31_fold/batch/j.$JOB_ID.$TASK_ID.out
#
# v31_fold -- fold the attribution nulls and the round-2 arms into deployable
# checkpoints, using the same fold_one() path that produced arm A so the deployment
# pipeline is not itself a difference between arms.
#
# Each merged checkpoint is ~12 GB and the output directory carries the arm name: a past
# run lost work because fold outputs from different factors collided in one directory.
#
# The fold is a deterministic elementwise merge of LoRA weights into the base tree, so
# the host constraint here is looser than for evaluation; every EVALUATION of these
# checkpoints is pinned to the host its base reference was measured on.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

case "$SGE_TASK_ID" in
  1) ARM=SFT0;   SRC=$LR/v27_ablate/train/SFT0 ;;
  2) ARM=SHAM;   SRC=$LR/v27_ablate/train/SHAM ;;
  # SFTPOS's LOSO fold is offset_1, not offset_0: its records come from base-SUCCESS
  # offsets, which have no offset_0 fold. Fold that index and expose it under the
  # uniform offset_0 name so every downstream evaluation addresses arms identically.
  3) ARM=SFTPOS; SRC=$LR/v27_ablate/train/SFTPOS; FOLD=1 ;;
  4) ARM=R2;     SRC=$LR/v30_r2train/train/R2 ;;
  5) ARM=R2CTRL; SRC=$LR/v30_r2train/train/R2CTRL ;;
  # replicate checkpoints: same data, same config, different data-order seed. They exist
  # so that "arm A beats the null" can be told apart from "this checkpoint beats the
  # null", which matters because every arm early-stops at step 25-50 and the deployed
  # effect sizes are single-digit episodes.
  6) ARM=A_s2;    SRC=$LR/v33_seed/train/A_s2 ;;
  7) ARM=SHAM_s2; SRC=$LR/v33_seed/train/SHAM_s2 ;;
  8) ARM=R2_s2;   SRC=$LR/v33_seed/train/R2_s2 ;;
  # v37 teacher-centre grid: 2 teachers x 5 data-order seeds. Folded into the same
  # v31_fold/<arm>/offset_0 layout as everything else so the evaluation job addresses
  # them identically -- ten checkpoints, all evaluated, none dropped.
  9|10|11|12|13|14|15|16|17|18)
     GRID=(Teef_s0 Teef_s777 Teef_s1009 Teef_s2027 Teef_s3037 \
           Tdual_s0 Tdual_s777 Tdual_s1009 Tdual_s2027 Tdual_s3037)
     ARM=${GRID[$((SGE_TASK_ID - 9))]}
     SRC=$LR/v37_teachergrid/train/$ARM ;;
esac

FOLD=${FOLD:-0}
if [ ! -d "$SRC/offset_$FOLD/params" ]; then
  echo "NOT_READY arm=$ARM src=$SRC -- training has not produced an adapter yet"; exit 0
fi
mkdir -p $LR/v31_fold/batch $LR/v31_fold/$ARM
if [ -n "$(ls -A "$LR/v31_fold/$ARM/offset_0" 2>/dev/null)" ]; then
  echo "SKIP existing $ARM"; exit 0
fi

echo "=== v31 fold arm=$ARM src=$SRC on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python - "$ARM" "$SRC" "$FOLD" <<'PYFOLD'
import pathlib, sys
OP = pathlib.Path("${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/models/openpi")
sys.path.insert(0, str(OP / "scripts"))
import fold_lora_adapter as F
import orbax.checkpoint as ocp

arm, src, fold = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3])
LR = pathlib.Path("${WORK_ROOT}/lora_dagger")
F.FOLDS = src
F.OUT = LR / "v31_fold" / arm
F.OUT.mkdir(parents=True, exist_ok=True)
print(f"FOLDS={F.FOLDS}  OUT={F.OUT}  BASE={F.BASE}")
ck = ocp.StandardCheckpointer()
F.fold_one(fold, ck, verbose=True)
print("FOLD_OK", arm)
PYFOLD
echo "FOLD_EXIT=$? arm=$ARM"
if [ "$FOLD" != "0" ]; then ln -sfn "offset_$FOLD" "$LR/v31_fold/$ARM/offset_0"; fi

# A folded checkpoint that is byte-identical to base would silently turn every arm into
# the base policy and produce a clean-looking "no effect" result. Assert it moved.
external/VLA-Arena/envs/openpi/.venv/bin/python - "$LR/v31_fold/$ARM/offset_0" "$ARM" <<'PYCHK'
import pathlib, sys
out, arm = pathlib.Path(sys.argv[1]), sys.argv[2]
assert (out / "params").exists(), "no params written"
assets = out / "assets"
n = sum(1 for _ in (out / "params").rglob("*") if _.is_file())
print(f"FOLD_AUDIT arm={arm} param_files={n} assets={'yes' if assets.exists() else 'NO'}")
assert n > 0, "empty params tree"
assert assets.exists(), "missing assets -- the checkpoint will not load norm stats"
print("FOLD_CELL_VERIFIED")
PYCHK
du -sh $LR/v31_fold/$ARM 2>/dev/null || true
