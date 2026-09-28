#!/usr/bin/env bash
# Review 2026-09-13. Build every record set the review experiments train on. CPU only, reads
# derived jsonl and writes symlinked roots, so it runs on the frontend. Idempotent: each
# builder removes its own destination first.
set -euo pipefail
LR=${WORK_ROOT}/lora_dagger
O=$LR/review0913
S=${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913
L=${SE_VLA_ROOT}/roundG/pi05_stage2/lora
cd ${SE_VLA_ROOT}
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python
export PYTHONPATH=src:external/VLA-Arena:.
mkdir -p $O/e1 $O/e5 $O/e6

echo "=== source checks ==="
for p in $LR/se_r2/s1s2_records/derived/folds/offset_0/train.jsonl \
         $LR/c1_collect/records/derived/folds/offset_0/train.jsonl \
         $LR/inloop_collect/records/derived/folds/offset_0/train.jsonl \
         $LR/r2_collect/records/derived/folds/offset_0/train.jsonl \
         $LR/se_pi0_q05/s1s2_records/derived/folds/offset_0/train.jsonl \
         $LR/se_dyn_L2t0/s1s2_records/derived/folds/offset_0/train.jsonl; do
  [ -s "$p" ] && echo "  OK $p ($(wc -l < "$p") rows)" || { echo "  MISSING $p"; exit 1; }
done
for d in $LR/se_r2/s1s2_records/blobs $LR/pi0_col_t2/records/blobs $LR/se_dyn_L2t0/s1s2_records/blobs; do
  [ -d "$d" ] && echo "  OK blobs $d" || { echo "  MISSING blobs $d"; exit 1; }
done

echo "=== E1: SFT0 / SHAM (verified builder from the v27 attribution study) ==="
$VENV $L/build_v27_ablation_records.py --src $LR/se_r2/s1s2_records --dst $O/e1/sft0_records --mode sft0
$VENV $L/build_v27_ablation_records.py --src $LR/se_r2/s1s2_records --dst $O/e1/sham_records --mode sham --seed 1234
for m in sft0 sham; do
  $VENV - "$LR/se_r2/s1s2_records" "$O/e1/${m}_records" "$m" <<"PY"
import sys, pathlib
sys.path.insert(0, "roundG/pi05_stage2/lora")
import build_v27_ablation_records as B
B.verify(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3])
PY
done
echo "=== E1: SFTPOS ==="
$VENV $S/build_sftpos_current.py $LR/se_r2/s1s2_records $O/e1/sftpos_records

echo "=== E5: round-1 observe-only vs in-loop (same collection checkpoint, same builder) ==="
$VENV $L/build_round_records.py --src-root $LR/c1_collect/records --dst-root $O/e5/oo_r1_records --mode s1s2
$VENV $L/build_round_records.py --src-root $LR/inloop_collect/records --dst-root $O/e5/il_r1_records --mode s1s2

echo "=== E6: round-2 replacement bank, and the accumulated bank subsampled to its size ==="
$VENV $L/build_round_records.py --src-root $LR/r2_collect/records --dst-root $O/e6/r2only_records --mode s1s2
N=$(wc -l < $O/e6/r2only_records/derived/folds/offset_0/train.jsonl)
echo "  r2only train rows = $N"
$VENV $S/subsample_accum.py $LR/se_r2/s1s2_records $O/e6/accum_matched_records "$N" 0

echo "=== built roots: rows, composition, blobs ==="
for r in $O/e1/sft0_records $O/e1/sham_records $O/e1/sftpos_records \
         $O/e5/oo_r1_records $O/e5/il_r1_records $O/e6/r2only_records $O/e6/accum_matched_records; do
  $VENV - "$r" <<"PY"
import json, sys, os, collections
r = sys.argv[1]
f = os.path.join(r, "derived/folds/offset_0/train.jsonl")
v = os.path.join(r, "derived/folds/offset_0/validation.jsonl")
c = collections.Counter(); n = 0
for line in open(f):
    x = json.loads(line); n += 1
    c["triggered" if x.get("triggered") else ("quiet" if x.get("quiet") else "neither")] += 1
nv = sum(1 for _ in open(v)) if os.path.exists(v) else 0
print("  %-60s train=%-6d %s val=%-4d blobs=%s" % (r.split("review0913/")[-1], n, dict(c), nv,
      os.path.isdir(os.path.join(r, "blobs"))))
PY
done
echo "BUILDS_DONE"
