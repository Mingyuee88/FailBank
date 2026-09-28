#!/usr/bin/env python3
"""Build a clean offset holdout: train on offsets 0-31, evaluate on 32-49.

Every number reported on L1t2 so far is in-sample. The collection walked offsets 0-47
(minus 9) and the evaluation walks 0-49, so 47 of 50 evaluated offsets were in the training
data -- 94% overlap. The three genuinely unseen ones (9, 48, 49) are too few to conclude
from, even though they happen to look fine.

Cross-TASK generalisation is already clean and holds up (pi_2 trained only on L1t2 scores
SR 85.0 / 88.0 on the unseen L1-T1 / L1-T4 against 89.0 on its training task). What is
missing is the easier check: same task, unseen initial states.

Rows carry an `offset` field, so the split is exact. Blobs are content-addressed and shared
by symlink -- no copying, and identical content resolves to the same path either way.
"""
import argparse, json, pathlib, os, collections

LR = pathlib.Path("${WORK_ROOT}/lora_dagger")
FOLD = "derived/folds/offset_0"

# Defaults reproduce the original static-suite split (se_r2 -> se_holdout, train on 0..31).
# --src/--dst/--train-max retarget it at another collection, e.g. the dynamic suite, where
# the same in-sample trap applies: collection walked offsets 0..47 and evaluation walks
# 0..49, so without a split the "generalisation" number is 94% memorisation.
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--src", default=str(LR / "se_r2" / "s1s2_records"))
_ap.add_argument("--dst", default=str(LR / "se_holdout" / "s1s2_records"))
_ap.add_argument("--train-max", type=int, default=32,
                 help="offsets [0, train_max) go to training; the rest are held out")
_args = _ap.parse_args()
SRC = pathlib.Path(_args.src)
DST = pathlib.Path(_args.dst)
TRAIN_OFFSETS = set(range(0, _args.train_max))
print("src=%s\ndst=%s\ntrain_offsets=[0,%d)" % (SRC, DST, _args.train_max))

(DST / FOLD).mkdir(parents=True, exist_ok=True)
kept = collections.Counter()
for split in ("train", "validation"):
    src = SRC / FOLD / (split + ".jsonl")
    if not src.is_file():
        print("  %s missing" % src); continue
    out = []
    seen_off = collections.Counter()
    for line in src.open():
        try: r = json.loads(line)
        except Exception: continue
        off = r.get("offset")
        if off is None:
            kept["no_offset"] += 1; continue
        seen_off[int(off)] += 1
        if int(off) in TRAIN_OFFSETS:
            out.append(line); kept[split] += 1
        else:
            kept[split + "_dropped"] += 1
    (DST / FOLD / (split + ".jsonl")).write_text("".join(out))
    print("  %-11s kept %5d / %5d rows   offsets in source: %d"
          % (split, kept[split], kept[split] + kept[split + "_dropped"], len(seen_off)))

link = DST / "blobs"
if not link.exists():
    os.symlink(SRC / "blobs", link)
print("  blobs symlinked -> %s" % (SRC / "blobs"))
print()
print("train offsets 0-31, evaluation will use 32-49 (18 offsets never seen in training)")
print("wrote %s" % DST)
