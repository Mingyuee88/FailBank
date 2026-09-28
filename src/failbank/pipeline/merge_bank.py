#!/usr/bin/env python3
"""Stage 3: accumulate the failure bank across rounds.

    failbank-merge-bank --round r1=<round1>/records --round r2=<round2>/records \
                        --out <bank> [--fold offset_0]

Each round's derived fold is ADDED to the bank, never substituted: round k harvests
failures the round k-1 policy did not make, so the bank grows monotonically. Records are
de-duplicated by ``record_id`` (first round wins) and tagged with ``bank_round``.

Blobs are content-addressed, so the blob stores merge by hard link with no duplication and
no risk of one round's observation shadowing another's. ``<bank>/episodes/<tag>`` are
symlinks to each round's episodes (provenance only; nothing downstream reads them).
"""
import argparse
import collections
import json
import os
import pathlib
import subprocess


def parse_rounds(values):
    rounds = []
    for v in values:
        tag, sep, path = v.partition("=")
        if not sep or not tag or not path:
            raise SystemExit(f"--round expects TAG=PATH, got {v!r}")
        rounds.append((tag, pathlib.Path(path)))
    tags = [t for t, _ in rounds]
    if len(set(tags)) != len(tags):
        raise SystemExit(f"duplicate round tags: {tags}")
    return rounds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--round", dest="rounds", action="append", required=True,
                    help="TAG=PATH of a round's records root; repeat in round order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", default="offset_0")
    a = ap.parse_args(argv)
    rounds = parse_rounds(a.rounds)
    out = pathlib.Path(a.out)
    fold = a.fold
    for tag, src in rounds:
        print(f"{tag}={src}")
        assert (src / "derived/folds" / fold / "train.jsonl").is_file(), f"missing derived fold in {src}"
    print(f"out={out}\nfold={fold}")

    if out.exists():
        subprocess.run(["rm", "-rf", str(out)], check=True)
    (out / "derived/folds" / fold).mkdir(parents=True)
    (out / "blobs").mkdir()
    for _, src in rounds:
        subprocess.run(["cp", "-aln", f"{src}/blobs/.", f"{out}/blobs/"], check=False)
    n_blob = sum(1 for p in (out / "blobs").rglob("*") if p.is_file())
    print(f"merged blob store: {n_blob} files")

    (out / "episodes").mkdir(exist_ok=True)
    for tag, src in rounds:
        link = out / "episodes" / tag
        if not link.exists():
            os.symlink(src.resolve() / "episodes", link)

    seen = set()
    counts = collections.Counter()
    stages = collections.Counter()
    for split in ("train.jsonl", "validation.jsonl"):
        rows = []
        for tag, src in rounds:
            f = src / "derived/folds" / fold / split
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
        with (out / "derived/folds" / fold / split).open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"{split}: {len(rows)} records")
    for extra in (rounds[0][1] / "derived").glob("*"):
        if extra.is_file():
            subprocess.run(["cp", str(extra), str(out / "derived" / extra.name)], check=False)
    print("composition:", dict(counts))
    print("triggered stages in merged train:", dict(stages))
    for tag, _ in rounds:
        assert counts[f"train.jsonl:{tag}"] > 0, f"round {tag} contributed no training records"
    assert stages.get("S1_early", 0) > 0, "no trainable S1 records in the merged bank"
    print("BANK_MERGED")


if __name__ == "__main__":
    main()
