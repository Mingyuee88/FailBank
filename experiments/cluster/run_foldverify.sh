#!/usr/bin/env bash
#$ -N FoldVerify
#$ -cwd
#$ -l gpu_card=1
#$ -pe smp 8
#$ -j y
#
# Does a refolded checkpoint equal the original NUMERICALLY? File comparison cannot answer
# this: orbax OCDBT names data files by content hash and stamps _CHECKPOINT_METADATA and
# manifest.ocdbt with write-time state, so those differ on every write regardless of content.
# Load both param trees and compare arrays.
#
# This is the load-bearing check for deleting ~1.25 TB of folded checkpoints: if refolding is
# not bit-identical, evaluations run on a refolded checkpoint cannot be tabulated with the
# historical numbers, and the deletion plan is void.
set -euo pipefail
export HF_HOME=${WORK_ROOT}/hf_cache
export JAX_PLATFORMS=cpu
cd ${SE_VLA_ROOT}
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python
REFOLD=${GROUP_ROOT}/tmp/foldtest/offset_0/params
ORIG=${WORK_ROOT}/lora_dagger/nocurr/ckpt/R2_q00/offset_0/params
echo "=== FoldVerify on $(hostname) ==="
/usr/bin/time -v $VENV - "$REFOLD" "$ORIG" <<"PYV" 2>&1
import pathlib, sys, numpy as np
OP = pathlib.Path("${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/models/openpi")
sys.path.insert(0, str(OP / "scripts"))
import fold_lora_adapter as F
a = F.flatten(F.restore_numpy(pathlib.Path(sys.argv[1])))
b = F.flatten(F.restore_numpy(pathlib.Path(sys.argv[2])))
ka, kb = set(a), set(b)
print("leaves: refold=%d orig=%d  only_refold=%d only_orig=%d" % (len(ka), len(kb), len(ka-kb), len(kb-ka)))
identical = differing = 0; worst = (0.0, None)
for k in sorted(ka & kb):
    x, y = np.asarray(a[k]), np.asarray(b[k])
    if x.shape != y.shape:
        differing += 1; print("  SHAPE differs:", k, x.shape, y.shape); continue
    if np.array_equal(x, y):
        identical += 1
    else:
        differing += 1
        d = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
        if d > worst[0]: worst = (d, k)
print("ARRAYS bit-identical: %d   differing: %d" % (identical, differing))
print("worst abs diff: %.3e  at %s" % (worst[0], worst[1]))
print("VERDICT:", "BIT-IDENTICAL -- refolding is lossless" if differing == 0 else "NOT identical -- see worst diff")
PYV
echo "FOLDVERIFY_DONE"
