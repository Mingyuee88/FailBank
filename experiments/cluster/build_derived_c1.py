#!/usr/bin/env python3
"""Build derived records from the C1 failure collection, staged by lead time to risk.

A thin wrapper over `src/vlsa_arena/learning_records/build_derived.py`. That module is left
untouched: it produced every published record set in this project, and its teacher-weighting,
outcome, aftermath and quiet definitions must stay bit-identical so the new records remain
commensurable with the old ones.

Two things change, both required by what this collection is FOR:

1. NO OUTCOME FILTER. The original drops an episode when `not success or pc != 0.0` -- which
   discards not just failures but any episode carrying policy cost. That single line is why
   all 2830 existing records have `eventual_success = True` and why the policy has never seen
   a crash. Applied to C1 it would drop all 47 episodes (every one carries cost; total 2124).

2. CURRICULUM STAGE. Each record is labelled with `steps_to_first_risk` -- its distance in
   steps from the episode's first crossing of `cost_pair_min_distance < 0.1087`, the
   gated-fallback threshold (crash recall 21/26, 12.5% FP, median lead 24 steps) -- and
   staged on it:

       S1_early      lead > 30      small corrections, inside support, learnable
       S2_mid        15 <= lead <= 30
       S3_emergency  0 <= lead < 15  NOT trained; handed to the fallback
       POST_crossing lead < 0        damage handling after contact; NOT trained
       NO_RISK       episode never crosses; unstageable -- the replay/anchor pool

   Measured on this collection, median |delta| rises monotonically across the stages
   (0.1818 / 0.2664 / 0.3764 / 0.5485), independently reproducing the support-distance
   gradient E11 measured on the shield's own corrections (loose 0.1887 -> tight 0.5970).
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, "${SE_VLA_ROOT}/src/vlsa_arena/learning_records")
import build_derived as B

RISK_THRESHOLD = 0.1087
STAGES = (("S1_early", 30, 10**9), ("S2_mid", 15, 30),
          ("S3_emergency", 0, 15), ("POST_crossing", -10**9, 0))


def first_risk_step(episode_dir: Path):
    """Index of the first step whose cost_pair distance is inside the risk threshold."""
    f = episode_dir / "raw_steps.jsonl"
    if not f.exists():
        return None
    for i, line in enumerate(f.open()):
        try:
            d = (json.loads(line).get("runtime_info") or {}).get("cost_pair_min_distance")
        except Exception:
            continue
        if d is not None and d < RISK_THRESHOLD:
            return i
    return None


def stage_for(lead):
    if lead is None:
        return "NO_RISK"
    for name, lo, hi in STAGES:
        if lo <= lead < hi:
            return name
    return "NO_RISK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-root", type=Path, required=True)
    a = ap.parse_args()
    root = a.records_root.resolve()
    th = B.Thresholds()

    all_rec, all_cand = [], []
    kept = dropped = 0
    stage_counts = Counter()
    for ej in sorted(root.glob("episodes/*/*/episode.json")):
        ed = ej.parent
        if not (ed / "COMPLETE").exists():
            dropped += 1
            continue
        offset, success, pc, _ = B.resolve_offset_success(ed)
        # deliberately NOT filtering on success/pc -- see module docstring
        recs, cands = B.build_episode(ed, offset, success, th)
        fr = first_risk_step(ed)
        for r in recs:
            lead = None if fr is None else fr - int(r["step_index"])
            r["steps_to_first_risk"] = lead
            r["curriculum_stage"] = stage_for(lead)
            r["risk_threshold"] = RISK_THRESHOLD
            stage_counts[r["curriculum_stage"]] += 1
        all_rec.extend(recs)
        all_cand.extend(cands)
        kept += 1

    print(f"episodes kept: {kept}  (incomplete dropped: {dropped})")
    print(f"total records: {len(all_rec)} | triggered: {sum(r['triggered'] for r in all_rec)}")
    print(f"eventual_success distribution: {Counter(r['eventual_success'] for r in all_rec)}")
    print(f"stages (all records): {dict(stage_counts)}")
    trig = Counter(r["curriculum_stage"] for r in all_rec if r["triggered"])
    print(f"stages (triggered only, i.e. carrying a teacher target): {dict(trig)}")
    print(f"triggered teacher buckets: "
          f"{dict(Counter(r['teacher']['bucket'] for r in all_rec if r['triggered']))}")

    assert kept > 0, "no episodes kept"
    assert any(not r["eventual_success"] for r in all_rec), \
        "no failure records survived -- the outcome filter is still active somewhere"
    assert trig.get("S1_early", 0) > 0, "no trainable S1 records -- the curriculum has no stage 1"

    offsets = sorted({r["offset"] for r in all_rec}, key=lambda x: int(x))
    derived = root / "derived"
    for ho in offsets:
        fr_dir = derived / "folds" / f"offset_{ho}"
        train = [r for r in all_rec if r["offset"] != ho]
        val = [r for r in all_rec if r["offset"] == ho]
        tids = {r["record_id"] for r in train}
        B.write_jsonl(fr_dir / "train.jsonl", train)
        B.write_jsonl(fr_dir / "validation.jsonl", val)
        B.write_jsonl(fr_dir / "train_critic_candidates.jsonl",
                      [c for c in all_cand if c["parent_record_id"] in tids])
        B.write_jsonl(fr_dir / "validation_critic_candidates.jsonl",
                      [c for c in all_cand if c["parent_record_id"] not in tids])
        fr_dir.mkdir(parents=True, exist_ok=True)
        (fr_dir / "manifest.json").write_text(json.dumps({
            "schema_version": B.SCHEMA, "heldout_offset": ho,
            "outcome_filter": "DISABLED -- failures are the point of this collection",
            "risk_threshold": RISK_THRESHOLD,
            "stage_counts_train": dict(Counter(r["curriculum_stage"] for r in train)),
            "train_records": len(train), "validation_records": len(val),
            "train_triggered": sum(r["triggered"] for r in train),
        }, indent=2, sort_keys=True))
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "schema_version.txt").write_text(B.SCHEMA + "\n")
    print(f"wrote {len(offsets)} LOSO folds under {derived}/folds/")
    print("C1_DERIVED_BUILT")


if __name__ == "__main__":
    main()
