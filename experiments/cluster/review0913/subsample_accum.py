#!/usr/bin/env python3
"""Review 2026-09-13, E6. Subsample an accumulated bank's train split to a target row count.

Stratified by (triggered, quiet) so the class composition of the accumulated bank is kept,
with a fixed seed. validation.jsonl and every other non-train file are copied unchanged, so
the guard batch is identical to the full accumulated bank.

usage: subsample_accum.py SRC_ROOT DST_ROOT TARGET_ROWS SEED
"""
import collections
import json
import os
import pathlib
import random
import shutil
import sys

src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
target, seed = int(sys.argv[3]), int(sys.argv[4])
if dst.exists():
    shutil.rmtree(dst)
(dst / "derived" / "folds").mkdir(parents=True)
for shared in ("blobs", "episodes"):
    if (src / shared).exists():
        os.symlink(src / shared, dst / shared)
for f in (src / "derived").glob("*"):
    if f.is_file():
        shutil.copy2(f, dst / "derived" / f.name)

for fold in sorted((src / "derived" / "folds").glob("*")):
    out_fold = dst / "derived" / "folds" / fold.name
    out_fold.mkdir()
    for f in fold.glob("*"):
        if f.name != "train.jsonl":
            shutil.copy2(f, out_fold / f.name)
            continue
        rows = [line for line in open(f) if line.strip()]
        assert target <= len(rows), "target %d exceeds source rows %d" % (target, len(rows))
        strata = collections.defaultdict(list)
        for line in rows:
            r = json.loads(line)
            strata[(bool(r.get("triggered")), bool(r.get("quiet")))].append(line)
        rng = random.Random(seed)
        keys = sorted(strata)
        quotas = {k: int(target * len(strata[k]) / len(rows)) for k in keys}
        # hand the rounding remainder to the largest strata so the total is exact
        for k in sorted(keys, key=lambda k: -len(strata[k]))[: target - sum(quotas.values())]:
            quotas[k] += 1
        out = []
        for k in keys:
            out += rng.sample(strata[k], quotas[k])
        rng.shuffle(out)
        with open(out_fold / f.name, "w") as fout:
            fout.writelines(out)
        comp = collections.Counter(
            (json.loads(l).get("triggered"), json.loads(l).get("quiet")) for l in out)
        src_comp = {k: len(v) for k, v in strata.items()}
        assert len(out) == target
        print("SUBSAMPLED fold=%s from=%d to=%d src_strata=%s out_strata=%s" % (
            fold.name, len(rows), len(out), src_comp, dict(comp)))
