"""Accumulate the bank: round-2 records are ADDED to round-1's, not substituted for them.

That accumulation is the point of the design -- each round harvests failures the previous
policy did not make, and the bank is supposed to grow monotonically. Blobs are
content-addressed, so the two stores merge by hard link with no duplication and no risk of
one round's observation shadowing another's.
"""
import json, os, pathlib, subprocess, collections
import argparse

LR = pathlib.Path("${WORK_ROOT}/lora_dagger")

# Defaults reproduce the original static merge exactly (c1_collect + r2_collect -> bank_r2).
# --r1/--r2/--out retarget it at another suite; the dynamic arm needs the same accumulation
# because a single round of 47 cells yields only 6 with policy-induced cost, against 26 on
# the static task -- the bank is meant to grow monotonically across rounds.
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--r1", default=str(LR / "c1_collect/records"))
_ap.add_argument("--r2", default=str(LR / "r2_collect/records"))
_ap.add_argument("--out", default=str(LR / "bank_r2"))
_ap.add_argument("--fold", default="offset_0")
_ap.add_argument("--tag1", default="r1")
_ap.add_argument("--tag2", default="r2")
_args = _ap.parse_args()
R1 = pathlib.Path(_args.r1)
R2 = pathlib.Path(_args.r2)
OUT = pathlib.Path(_args.out)
FOLD = _args.fold
print(f"r1={R1}\nr2={R2}\nout={OUT}\nfold={FOLD}")
for _p in (R1, R2):
    assert (_p / "derived/folds" / FOLD / "train.jsonl").is_file(), f"missing derived fold in {_p}"

if OUT.exists():
    subprocess.run(["rm", "-rf", str(OUT)], check=True)
(OUT / "derived/folds" / FOLD).mkdir(parents=True)
(OUT / "blobs").mkdir()
for src in (R1, R2):
    subprocess.run(["cp", "-aln", f"{src}/blobs/.", f"{OUT}/blobs/"], check=False)
n_blob = sum(1 for _ in (OUT / "blobs").rglob("*") if _.is_file())
print(f"merged blob store: {n_blob} files")

# episodes are needed by nothing downstream but keep provenance discoverable
(OUT / "episodes").mkdir(exist_ok=True)
for src, tag in ((R1, _args.tag1), (R2, _args.tag2)):
    link = OUT / "episodes" / tag
    if not link.exists():
        os.symlink(src / "episodes", link)

seen = set()
counts = collections.Counter()
stages = collections.Counter()
for split in ("train.jsonl", "validation.jsonl"):
    rows = []
    for src, tag in ((R1, _args.tag1), (R2, _args.tag2)):
        f = src / "derived/folds" / FOLD / split
        if not f.exists():
            continue
        for line in f.open():
            r = json.loads(line)
            rid = r.get("record_id")
            if rid in seen:
                counts["duplicate_skipped"] += 1
                continue
            seen.add(rid)
            r["bank_round"] = tag
            rows.append(r)
            counts[f"{split}:{tag}"] += 1
            if split == "train.jsonl" and (r.get("formal_projection") or {}).get("triggered"):
                stages[r.get("curriculum_stage")] += 1
    with (OUT / "derived/folds" / FOLD / split).open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"{split}: {len(rows)} records")
for extra in (R1 / "derived").glob("*"):
    if extra.is_file():
        subprocess.run(["cp", str(extra), str(OUT / "derived" / extra.name)], check=False)
print("composition:", dict(counts))
print("triggered stages in merged train:", dict(stages))
assert counts[f"train.jsonl:{_args.tag1}"] > 0 and counts[f"train.jsonl:{_args.tag2}"] > 0, "a round is missing"
assert stages.get("S1_early", 0) > 0, "no trainable S1 records in the merged bank"
print("BANK_MERGED")
