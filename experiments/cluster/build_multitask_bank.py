#!/usr/bin/env python3
"""Build a four-task training bank from the per-task collections.

The current best recipe -- no staging, quiet_weight 0, 800 steps single phase -- reaches
SR 90.8 / CC 18.5, but it was only ever trained and evaluated on L1t2. Whether that is a
property of the recipe or of that one task is untested, and it is the most valuable open
question left now that the safety layer is closed.

Blobs are content-addressed, so identical content lands on the same relative path in every
bank and the merged store can hard-link them without collision or duplication. The derived
train/validation jsonl files are concatenated; each row already carries its own blob paths.
"""
import json, os, pathlib, sys, collections

LR = pathlib.Path("${WORK_ROOT}/lora_dagger")
SRC = [("L1t2", LR/"se_r2"), ("L1t1", LR/"mcol_t1"), ("L1t3", LR/"mcol_t3"), ("L1t4", LR/"mcol_t4")]
DST = LR / "se_multi"
FOLD = "derived/folds/offset_0"

def main():
    missing = [n for n, p in SRC if not (p/"s1s2_records"/FOLD/"train.jsonl").is_file()]
    if missing:
        print("record sets not built yet for: %s" % missing)
        print("run build_derived_c1.py + build_round_records.py on those banks first")
        return 1
    (DST/"s1s2_records"/FOLD).mkdir(parents=True, exist_ok=True)
    blobs = DST/"s1s2_records"/"blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    linked = collections.Counter()
    for split in ("train", "validation"):
        rows = []
        for name, root in SRC:
            f = root/"s1s2_records"/FOLD/(split + ".jsonl")
            if not f.is_file(): continue
            n = 0
            for line in f.open():
                rows.append(line); n += 1
            print("  %-6s %-10s %6d rows" % (name, split, n))
        (DST/"s1s2_records"/FOLD/(split + ".jsonl")).write_text("".join(rows))
        print("  merged %-10s %6d rows" % (split, len(rows)))
    # hard-link every blob from every source bank
    for name, root in SRC:
        src_blobs = root/"s1s2_records"/"blobs"
        if not src_blobs.is_dir():
            print("  %s has no blobs dir -- rows reference the collection bank directly" % name)
            continue
        for p in src_blobs.rglob("*"):
            if not p.is_file(): continue
            rel = p.relative_to(src_blobs)
            q = blobs/rel
            if q.exists(): linked["dup"] += 1; continue
            q.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(p, q); linked["linked"] += 1
            except OSError:
                import shutil; shutil.copy2(p, q); linked["copied"] += 1
    print("  blobs: %s" % dict(linked))
    print()
    print("merged bank at %s" % DST)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
