#!/bin/bash
#$ -N R5Chain
#$ -cwd
#$ -q gpu@HOST_A
#$ -l gpu_card=1
#$ -pe smp 4
#$ -j y
# B_5 = B_4 u D(pi_4), then train round 5 at the same fixed quiet_weight=0.2.
# Fresh bank path (bank_r5d) -- bank paths are never reused after an interrupted merge.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
cd ${SE_VLA_ROOT}
L=${WORK_ROOT}/lora_dagger
OP=external/VLA-Arena/vla_arena/models/openpi
V=external/VLA-Arena/envs/openpi/.venv/bin/python
ORIG_BASE=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
export PYTHONPATH=src:external/VLA-Arena:.
STEPS=800; BATCH=32; QW=0.2; ARM=R5_q20
SEC=$L/round_curve

echo "=== step 1: build r5 record set ==="
$V roundG/pi05_stage2/lora/build_derived_c1.py --records-root "$L/r5_collect/records"
for MODE in s1 s1s2; do
  $V roundG/pi05_stage2/lora/build_round_records.py \
    --src-root "$L/r5_collect/records" --dst-root "$L/se_r5/${MODE}_records" --mode "$MODE"
done
echo "R5SET_DONE rows=$(wc -l < $L/se_r5/s1s2_records/derived/folds/offset_0/train.jsonl)"

echo "=== step 2: B_5 = B_4 u se_r5 ==="
$V roundG/pi05_stage2/lora/merge_bank.py \
  --r1 "$L/bank_r4d" --r2 "$L/se_r5/s1s2_records" \
  --out "$L/bank_r5d" --fold offset_0 --tag1 r1234 --tag2 r5
SRC=$L/bank_r5d
echo "B5_DONE rows=$(wc -l < $SRC/derived/folds/offset_0/train.jsonl)"

echo "=== step 3: train $ARM ==="
A1=$SEC/adapter/${ARM}; F1=$SEC/ckpt/${ARM}; C1=$F1/offset_0
if [ ! -s "$A1/offset_0/metrics.json" ]; then
  SE_VLA_TRAIN_BASE_CHECKPOINT="$ORIG_BASE" $V $OP/scripts/train_phase2_lora.py \
    --offset 0 --records-root "$SRC/blobs" --derived-root "$SRC/derived/folds" \
    --batch-size "$BATCH" --quiet-weight "$QW" \
    --smoke-steps "$STEPS" --validation-interval "$STEPS" --patience 99 \
    --output-root "$A1"
fi
$V - "$A1" "$ARM" <<'PYV'
import json, sys, pathlib
out, arm = sys.argv[1], sys.argv[2]
m = json.load(open(pathlib.Path(out) / "offset_0/metrics.json"))
print(f"GUARD {arm}: flow_ratio={m.get('best_quiet_flow_ratio')} "
      f"drift={m.get('best_quiet_action_drift')} accepted={m.get('accepted')}")
PYV
if ! $V -c "
import json,sys,pathlib
m=json.load(open(pathlib.Path('$A1')/'offset_0/metrics.json'))
sys.exit(0 if m.get('accepted') else 1)"; then
  echo "RC_REJECTED arm=$ARM -- round 5 is the curve's ceiling"
  exit 0
fi
if [ ! -d "$C1/params" ]; then
  $V - "$A1" "$F1" "$ORIG_BASE" <<'PYFOLD'
import pathlib, sys
OP = pathlib.Path("${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/models/openpi")
sys.path.insert(0, str(OP / "scripts"))
import fold_lora_adapter as F
import orbax.checkpoint as ocp
F.FOLDS = pathlib.Path(sys.argv[1]); F.OUT = pathlib.Path(sys.argv[2])
F.BASE = pathlib.Path(sys.argv[3]).resolve()
print("FOLD_BASE=%s" % F.BASE)
F.OUT.mkdir(parents=True, exist_ok=True)
F.fold_one(0, ocp.StandardCheckpointer(), verbose=True)
print("FOLD_OK")
PYFOLD
fi
[ -e "$C1/assets" ] || ln -s "$ORIG_BASE/assets" "$C1/assets"
$V roundG/pi05_stage2/lora/verify_family.py "$C1/params" pi05
echo "RC_DONE arm=$ARM"
