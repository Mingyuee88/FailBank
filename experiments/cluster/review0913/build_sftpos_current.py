#!/usr/bin/env python3
"""Review 2026-09-13, E1. SFTPOS null for the CURRENT recipe.

Behaviour cloning on the successful episodes of the main static round-2 bank
(se_r2/s1s2_records): the shield residual is removed from every record, only train records
from eventually-successful episodes are kept, and every kept triggered record gets teacher
weight 1.0. Train with --quiet-weight 1.0 so quiet and triggered success steps weigh equally.

The validation split is transformed exactly the way build_v27_ablation_records.py transforms
it for SFT0 (residual removed) and is NOT filtered, so this arm's guard batch matches SFT0's.
The residual definition is shared by importing clip_nominal from that script.

usage: build_sftpos_current.py SRC_ROOT DST_ROOT
"""
import json
import math
import os
import pathlib
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_v27_ablation_records import clip_nominal  # noqa: E402

src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
if dst.exists():
    shutil.rmtree(dst)
(dst / "derived" / "folds").mkdir(parents=True)
for shared in ("blobs", "episodes"):
    if (src / shared).exists():
        os.symlink(src / shared, dst / shared)
for f in (src / "derived").glob("*"):
    if f.is_file():
        shutil.copy2(f, dst / "derived" / f.name)

stats = dict(train_in=0, train_kept=0, triggered_kept=0, quiet_kept=0, validation=0)
for fold in sorted((src / "derived" / "folds").glob("*")):
    out_fold = dst / "derived" / "folds" / fold.name
    out_fold.mkdir()
    for f in fold.glob("*"):
        if f.suffix != ".jsonl":
            shutil.copy2(f, out_fold / f.name)
            continue
        is_train = f.name == "train.jsonl"
        with open(f) as fin, open(out_fold / f.name, "w") as fout:
            for line in fin:
                rec = json.loads(line)
                if is_train:
                    stats["train_in"] += 1
                    if not rec.get("eventual_success"):
                        continue
                if "executed_action" in rec and "nominal_action" in rec:
                    clipped = clip_nominal(rec["nominal_action"])
                    executed = list(rec["executed_action"])
                    executed[:3] = clipped[:3]
                    rec["executed_action"] = executed
                if is_train:
                    stats["train_kept"] += 1
                    if rec.get("triggered"):
                        rec["teacher"] = dict(rec.get("teacher") or {})
                        rec["teacher"]["weight"] = 1.0
                        stats["triggered_kept"] += 1
                    elif rec.get("quiet"):
                        stats["quiet_kept"] += 1
                elif f.name == "validation.jsonl":
                    stats["validation"] += 1
                fout.write(json.dumps(rec) + "\n")

# Verify the arm is what it claims, off the file the trainer will read.
train = dst / "derived" / "folds" / "offset_0" / "train.jsonl"
n = 0
for line in open(train):
    r = json.loads(line)
    n += 1
    assert r.get("eventual_success"), "a failure-episode record leaked into SFTPOS"
    cl = clip_nominal(r["nominal_action"])
    residual = math.sqrt(sum((r["executed_action"][i] - cl[i]) ** 2 for i in range(3)))
    assert residual <= 1e-9, "shield residual left in an SFTPOS record (%g)" % residual
    assert r.get("triggered") or r.get("quiet"), "record in neither weight class"
    if r.get("triggered"):
        assert r["teacher"]["weight"] == 1.0, "triggered record weight not reset to 1"
assert n == stats["train_kept"] and n > 0
print("SFTPOS_BUILT", stats, "verified_rows=%d" % n)
