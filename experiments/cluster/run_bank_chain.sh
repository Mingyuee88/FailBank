#!/bin/bash
#$ -N BankChain
#$ -cwd
#$ -q long
#$ -pe smp 4
#$ -j y
# Build the ACCUMULATED banks for rounds 3 and 4.
# The existing se_r3 was built from r3_collect ALONE (3364 rows, no bank_round tag), which
# is not B_3 = B_2 u D(pi_2). Rounds beyond 2 must accumulate or the round curve confounds
# "more rounds" with "switched to non-accumulating data".
set -euo pipefail
cd ${SE_VLA_ROOT}
L=${WORK_ROOT}/lora_dagger
V=external/VLA-Arena/envs/openpi/.venv/bin/python
export PYTHONPATH=src:external/VLA-Arena:.

echo "=== step 1: B_3 = bank_r2 u se_r3 ==="
rm -rf "$L/bank_r3c"
$V roundG/pi05_stage2/lora/merge_bank.py \
  --r1 "$L/bank_r2" --r2 "$L/se_r3/s1s2_records" \
  --out "$L/bank_r3c" --fold offset_0 --tag1 r12 --tag2 r3
echo "B3_DONE rows=$(wc -l < $L/bank_r3c/derived/folds/offset_0/train.jsonl)"

echo "=== step 2: annotate + build r4 record set ==="
$V roundG/pi05_stage2/lora/build_derived_c1.py --records-root "$L/r4_collect/records"
for MODE in s1 s1s2; do
  $V roundG/pi05_stage2/lora/build_round_records.py \
    --src-root "$L/r4_collect/records" --dst-root "$L/se_r4/${MODE}_records" --mode "$MODE"
done
echo "R4SET_DONE rows=$(wc -l < $L/se_r4/s1s2_records/derived/folds/offset_0/train.jsonl)"

echo "=== step 3: B_4 = B_3 u se_r4 ==="
rm -rf "$L/bank_r4"
$V roundG/pi05_stage2/lora/merge_bank.py \
  --r1 "$L/bank_r3c" --r2 "$L/se_r4/s1s2_records" \
  --out "$L/bank_r4" --fold offset_0 --tag1 r123 --tag2 r4
echo "B4_DONE rows=$(wc -l < $L/bank_r4/derived/folds/offset_0/train.jsonl)"
echo "CHAIN_OK"
