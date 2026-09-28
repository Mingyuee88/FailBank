#!/usr/bin/env bash
#$ -N RvClean
#$ -cwd
#$ -pe smp 1
#$ -j y
# Review 2026-09-13. Delete folded checkpoints that were created for the review experiments,
# once every evaluation that loads them has finished (enforced by -hold_jid at submission).
# LIST: one checkpoint directory per line.
#
# A path is deleted only if ALL of these hold, otherwise it is skipped and reported:
#   1. it matches the allowlist: review0913/e<N>/ckpt/<arm>, or exactly one of the four
#      E0 refolds nocurr{,_s1,_s2,_s3}/ckpt/R2_q00
#   2. it is a real directory (not a symlink) containing offset_0/params
#   3. the adapter it was folded from exists with offset_0/params, so it can be refolded
#      (refolding verified bit-identical: job 1421679, 51/51 leaves, max abs diff 0)
set -uo pipefail
: "${LIST:?set LIST}"
LR=${WORK_ROOT}/lora_dagger
freed=0; deleted=0; skipped=0
echo "=== RvClean list=$LIST host=$(hostname) $(date) ==="
df -h ${GROUP_ROOT} | tail -1
while read -r CK; do
  [ -z "$CK" ] && continue
  case "$CK" in
    $LR/review0913/e[0-9]/ckpt/*)
      rest=${CK#$LR/review0913/e?/ckpt/}
      if [[ ! "$rest" =~ ^[A-Za-z0-9_]+$ ]]; then echo "SKIP not_allowlisted $CK"; skipped=$((skipped+1)); continue; fi
      AD=${CK/\/ckpt\//\/adapter\/} ;;
    $LR/nocurr/ckpt/R2_q00|$LR/nocurr_s1/ckpt/R2_q00|$LR/nocurr_s2/ckpt/R2_q00|$LR/nocurr_s3/ckpt/R2_q00)
      AD=${CK/\/ckpt\//\/adapter\/} ;;
    *) echo "SKIP not_allowlisted $CK"; skipped=$((skipped+1)); continue ;;
  esac
  if [ -L "$CK" ]; then echo "SKIP symlink $CK"; skipped=$((skipped+1)); continue; fi
  if [ ! -d "$CK/offset_0/params" ]; then echo "SKIP no_folded_params $CK"; skipped=$((skipped+1)); continue; fi
  if [ ! -d "$AD/offset_0/params" ]; then echo "SKIP no_adapter_cannot_refold $CK (adapter $AD)"; skipped=$((skipped+1)); continue; fi
  sz=$(du -sb "$CK" 2>/dev/null | cut -f1)
  rm -rf "$CK"
  if [ -e "$CK" ]; then echo "FAILED rm $CK"; skipped=$((skipped+1)); continue; fi
  rmdir "$(dirname "$CK")" 2>/dev/null || true
  freed=$((freed + ${sz:-0})); deleted=$((deleted+1))
  echo "DELETED $CK ($(( ${sz:-0} / 1073741824 )) G); adapter kept at $AD"
done < "$LIST"
echo "CLEAN_DONE deleted=$deleted skipped=$skipped freed=$(( freed / 1073741824 )) G"
df -h ${GROUP_ROOT} | tail -1
