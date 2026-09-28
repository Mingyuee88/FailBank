#!/usr/bin/env bash
# Review 2026-09-13. Submit one cleanup job per experiment group, each held on every
# evaluation job that loads the checkpoints it deletes.
set -euo pipefail
S=${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913
LR=${WORK_ROOT}/lora_dagger
O=$LR/review0913
J=${SE_VLA_ROOT}/joblogs
IDS=$O/lists/job_ids.txt
cd ${SE_VLA_ROOT}

# Cleanup is pure file I/O: use a CPU queue if the cluster has one, else the GPU queue.
if qconf -sql 2>/dev/null | grep -qx long; then
  QARGS=(-q long)
else
  QARGS=(-q "gpu@HOST_A,gpu@HOST_B,gpu@HOST_C,gpu@HOST_D" -l gpu_card=1)
fi
echo "cleanup queue args: ${QARGS[*]}"

printf "%s\n" $LR/nocurr/ckpt/R2_q00 $LR/nocurr_s1/ckpt/R2_q00 $LR/nocurr_s2/ckpt/R2_q00 $LR/nocurr_s3/ckpt/R2_q00 > $O/lists/cleanup_e0.txt
printf "%s\n" $O/e1/ckpt/sft0_s1 $O/e1/ckpt/sham_s1 $O/e1/ckpt/sftpos_s1 > $O/lists/cleanup_e1s1.txt
printf "%s\n" $O/e1/ckpt/sft0_s2 $O/e1/ckpt/sft0_s3 $O/e1/ckpt/sham_s2 $O/e1/ckpt/sham_s3 $O/e1/ckpt/sftpos_s2 $O/e1/ckpt/sftpos_s3 > $O/lists/cleanup_e1s23.txt
printf "%s\n" $O/e5/ckpt/oo_r1 $O/e5/ckpt/il_r1 $O/e6/ckpt/r2only $O/e6/ckpt/accum_matched > $O/lists/cleanup_e56.txt
printf "%s\n" $O/e9/ckpt/dyn_r1_700 > $O/lists/cleanup_e9.txt

sub() {
  local name=$1 hold=$2 list=$3
  local out; out=$(qsub -terse -N RvClean "${QARGS[@]}" -o $J -hold_jid "$hold" -v LIST=$list $S/cleanup_ckpt.sh)
  local jid=${out%%.*}
  echo "$name $jid" | tee -a "$IDS"
}
# E0 refolds are loaded by E2 (8 arrays), E3 (2) and E7 (3)
sub CLEAN_E0    1440211,1440212,1440213,1440214,1440215,1440216,1440217,1440218,1440219,1440220,1440221,1440222,1440223 $O/lists/cleanup_e0.txt
# E1 seed-1 checkpoints: L2 apple (3) + L1 T1/T4 (6)
sub CLEAN_E1S1  1440231,1440232,1440233,1440242,1440243,1440244,1440245,1440246,1440247 $O/lists/cleanup_e1s1.txt
# E1 seed-2/3 checkpoints: L2 apple (6)
sub CLEAN_E1S23 1440236,1440237,1440238,1440239,1440240,1440241 $O/lists/cleanup_e1s23.txt
# E5/E6 checkpoints: apple + T1 + T4 for four arms (12)
sub CLEAN_E56   1440250,1440251,1440252,1440253,1440254,1440255,1440256,1440257,1440258,1440259,1440260,1440261 $O/lists/cleanup_e56.txt
# E9 checkpoint: dyn L2-T0 eval
sub CLEAN_E9    1440228 $O/lists/cleanup_e9.txt
echo "CLEANUP_SUBMITTED"
