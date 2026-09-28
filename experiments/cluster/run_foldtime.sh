#!/usr/bin/env bash
#$ -N FoldTime
#$ -cwd
#$ -l gpu_card=1
#$ -pe smp 8
#$ -j y
#
# Time ONE refold and verify fidelity, before deciding how much of the ~1.25 TB of folded
# checkpoints under lora_dagger can be deleted. Folding is pure numpy on CPU; the gpu_card
# request is only to land on a node with enough RAM (base + adapter + result in memory).
#
# Subject: nocurr/adapter/R2_q00 -> temp dir, compared against the EXISTING
# nocurr/ckpt/R2_q00/offset_0 that the same adapter produced originally.
set -euo pipefail
export HF_HOME=${WORK_ROOT}/hf_cache
export JAX_PLATFORMS=cpu
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python
ADAPTER=$LR/nocurr/adapter/R2_q00
EXISTING=$LR/nocurr/ckpt/R2_q00/offset_0
OUT=${GROUP_ROOT}/tmp/foldtest

[ -d "$ADAPTER/offset_0/params" ] || { echo "ABORT: no adapter params at $ADAPTER"; exit 1; }
[ -d "$EXISTING/params" ]         || { echo "ABORT: no existing ckpt at $EXISTING"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
echo "=== FoldTime on $(hostname) ==="
echo "adapter : $(du -sh $ADAPTER | cut -f1)"
echo "existing: $(du -sh $EXISTING | cut -f1)"
df -h ${GROUP_ROOT} | tail -1

/usr/bin/time -v $VENV - "$ADAPTER" "$OUT" <<"PYFOLD" 2> "$OUT/time.txt"
import pathlib, sys, time
OP = pathlib.Path("${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/models/openpi")
sys.path.insert(0, str(OP / "scripts"))
import fold_lora_adapter as F
import orbax.checkpoint as ocp
F.FOLDS = pathlib.Path(sys.argv[1]); F.OUT = pathlib.Path(sys.argv[2])
F.OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
F.fold_one(0, ocp.StandardCheckpointer(), verbose=True)
print("FOLD_SECONDS=%.1f" % (time.time() - t0))
PYFOLD
echo "--- timing / memory ---"
grep -E "FOLD_SECONDS|Elapsed \(wall|Maximum resident" "$OUT/time.txt" || true
grep -h "FOLD_SECONDS" "$OUT/time.txt" 2>/dev/null || true

echo "--- fidelity vs the existing checkpoint ---"
$VENV - "$OUT/offset_0/params" "$EXISTING/params" <<"PYCMP"
import sys, os, hashlib
a, b = sys.argv[1], sys.argv[2]
def index(root):
    d = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            d[os.path.relpath(p, root)] = os.path.getsize(p)
    return d
ia, ib = index(a), index(b)
only_a = sorted(set(ia) - set(ib)); only_b = sorted(set(ib) - set(ia))
same_size = [k for k in ia if k in ib and ia[k] == ib[k]]
diff_size = [k for k in ia if k in ib and ia[k] != ib[k]]
print("files: refolded=%d existing=%d  size-identical=%d  size-differs=%d" % (len(ia), len(ib), len(same_size), len(diff_size)))
print("only in refolded:", only_a[:5], "only in existing:", only_b[:5])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""): h.update(c)
    return h.hexdigest()[:16]
sample = sorted(same_size, key=lambda k: -ia[k])[:3]
for k in sample:
    print("  %-50s %s  %s  %s" % (k[:50], sha(os.path.join(a,k)), sha(os.path.join(b,k)),
          "MATCH" if sha(os.path.join(a,k)) == sha(os.path.join(b,k)) else "DIFFER"))
PYCMP
echo "--- sizes ---"; du -sh "$OUT/offset_0" "$EXISTING"
echo "FOLDTIME_DONE"
