#!/usr/bin/env bash
# Guard test for cleanup_ckpt.sh, then submit the held cleanup jobs only if the test passes.
set -euo pipefail
S=${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913
LR=${WORK_ROOT}/lora_dagger
O=$LR/review0913
B05=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
PI0CK=$LR/se_pi0_q05/ckpt_pi0fold_1405748
ADP=$LR/nocurr/adapter/R2_q00
T=$(mktemp -d ${GROUP_ROOT}/tmp/rvclean.XXXX)
ln -sfn $PI0CK $T/symlink_ckpt
cat > $T/neg_list.txt <<NEG
$B05
$PI0CK
$T/symlink_ckpt
$O/e1/ckpt/../../../nocurr/adapter
$O/e1/ckpt/sft0_s1/
$ADP
$O/e1/ckpt/sft0_s1
$LR/nocurr/ckpt/R2_q00
NEG
echo "=== GUARD TEST: every line must be SKIPPED ==="
before=$(du -sb $B05 $PI0CK $ADP | awk '{s+=$1} END {print s}')
LIST=$T/neg_list.txt bash $S/cleanup_ckpt.sh | grep -E "SKIP|DELETED|FAILED|CLEAN_DONE"
after=$(du -sb $B05 $PI0CK $ADP | awk '{s+=$1} END {print s}')
if [ ! -d $B05/params ] || [ ! -d $PI0CK ] || [ ! -d $ADP/offset_0/params ]; then echo "GUARD_TEST_FAILED: protected path missing"; exit 1; fi
if [ "$before" != "$after" ]; then echo "GUARD_TEST_FAILED: protected bytes changed $before -> $after"; exit 1; fi
if LIST=$T/neg_list.txt bash $S/cleanup_ckpt.sh | grep -q "^DELETED"; then echo "GUARD_TEST_FAILED: something was deleted"; exit 1; fi
echo "GUARD_TEST_PASSED protected_bytes_unchanged=$after"
rm -rf $T
echo "=== submitting cleanup jobs ==="
bash $S/submit_cleanup.sh
echo "=== RvClean in queue ==="
qstat -u USER 2>/dev/null | awk 'NR>2 && $3=="RvClean"{print $1, $3, $5}'
