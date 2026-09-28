#!/usr/bin/env python3
"""Build the two missing null arms for the shield-distillation claim.

The claim under test is "distilling the SHIELD'S CORRECTIONS improves the policy".
Arm A shows a gain over base, but nothing so far separates three explanations:

  (1) the shield's corrections carry useful information            <- the claim
  (2) touching the LoRA weights at all on this state distribution helps
  (3) any perturbation of matched magnitude helps

The teacher signal turns out to decompose exactly. For every record,

    executed_action = clip(nominal_action) + projection_delta

and `projection_delta` is nonzero on precisely the records whose
`formal_projection.triggered` is true -- verified: 866/2830 in the arm-A fold, and
`exec != clip(nominal)` matches `triggered` count for count. So the shield's
contribution can be surgically removed or scrambled while holding EVERYTHING else
fixed: same episodes, same states, same observations, same record count, same quiet
flags, same optimizer, same host.

  SFT0  projection_delta := 0
        The target becomes the base policy's own (clipped) action, i.e. pure
        self-distillation. Isolates explanation (2).

  SHAM  projection_delta := random direction, SAME magnitude, per record
        Isolates explanation (3). This is the same control that already killed the
        external-memory line of this project: shuffling the key->residual pairing
        there did not degrade performance, which proved the memory's CONTENT
        contributed nothing. If SHAM matches arm A here, the shield's content
        contributes nothing either and the honest reading is that the method is a
        magnitude-scheduled perturbation, not distilled safety knowledge.

Rotation is applied in the 3-D translation subspace only, because that is the only
subspace the CBF projection ever writes to. Gripper and rotation channels are copied
untouched, so SHAM cannot differ from A anywhere the shield did not act.

Blobs (observations, nominal chunks) are symlinked, not copied: they are identical by
construction and duplicating them would cost tens of GB per arm.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import shutil

MAX_TRANSLATION = 1.0  # SE_VLA_AEGIS_MAX_TRANSLATION default used in every collection


def clip_nominal(nominal):
    out = list(nominal)
    for i in range(3):
        out[i] = max(-MAX_TRANSLATION, min(MAX_TRANSLATION, float(out[i])))
    return out


def random_unit3(rng):
    """Uniform on the sphere via normal sampling (rejects the degenerate zero draw)."""
    while True:
        v = [rng.gauss(0.0, 1.0) for _ in range(3)]
        n = math.sqrt(sum(x * x for x in v))
        if n > 1e-9:
            return [x / n for x in v]


def transform(record, mode, rng):
    nominal = record["nominal_action"]
    executed = record["executed_action"]
    clipped = clip_nominal(nominal)
    delta = [float(executed[i]) - clipped[i] for i in range(3)]
    norm = math.sqrt(sum(d * d for d in delta))
    touched = norm > 1e-9

    new = list(executed)
    if mode == "sft0":
        new[:3] = clipped[:3]
    elif mode == "sham":
        if touched:
            u = random_unit3(rng)
            new[:3] = [clipped[i] + norm * u[i] for i in range(3)]
    else:
        raise ValueError(mode)
    record["executed_action"] = new
    return touched, norm


def build(src_root: pathlib.Path, dst_root: pathlib.Path, mode: str, seed: int,
          exclude_offsets: set[str] | None = None):
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)

    # observations are byte-identical across arms; link rather than copy
    for shared in ("blobs", "episodes"):
        s = src_root / shared
        if s.exists():
            os.symlink(s, dst_root / shared)

    rng = random.Random(seed)
    stats = {"folds": 0, "records": 0, "touched": 0, "norm_sum": 0.0, "norm_sum_out": 0.0}
    (dst_root / "derived").mkdir()
    src_derived = src_root / "derived"
    for extra in src_derived.glob("*"):
        if extra.is_file():
            shutil.copy2(extra, dst_root / "derived" / extra.name)
    (dst_root / "derived" / "folds").mkdir()

    for fold in sorted((src_derived / "folds").glob("*")):
        out_fold = dst_root / "derived" / "folds" / fold.name
        out_fold.mkdir()
        stats["folds"] += 1
        for f in fold.glob("*"):
            if f.suffix != ".jsonl":
                shutil.copy2(f, out_fold / f.name)
                continue
            with open(f) as fin, open(out_fold / f.name, "w") as fout:
                for line in fin:
                    rec = json.loads(line)
                    # The successful-trajectory SFT null draws from base-SUCCESS offsets,
                    # 18 of which are the held-out retention probe. Dropping them here
                    # keeps every arm's training set disjoint from that probe, so the
                    # arms stay comparable on it.
                    if exclude_offsets and str(rec.get("offset")) in exclude_offsets:
                        stats["excluded"] = stats.get("excluded", 0) + 1
                        continue
                    if "executed_action" in rec and "nominal_action" in rec:
                        touched, norm = transform(rec, mode, rng)
                        if f.name == "train.jsonl":
                            stats["records"] += 1
                            if touched:
                                stats["touched"] += 1
                                stats["norm_sum"] += norm
                                out_delta = math.sqrt(sum(
                                    (rec["executed_action"][i] - clip_nominal(rec["nominal_action"])[i]) ** 2
                                    for i in range(3)))
                                stats["norm_sum_out"] += out_delta
                    fout.write(json.dumps(rec) + "\n")
    return stats


def verify(src_root: pathlib.Path, dst_root: pathlib.Path, mode: str):
    """Re-read both roots and assert the transform did exactly what it claims.

    Four times in this project a knob has silently done nothing and the null result was
    read as a finding. The specific hazard here is producing an ablation root that is
    byte-identical to the original, which would make the ablation trivially "match" the
    method and look like a decisive negative result.
    """
    src_f = sorted((src_root / "derived" / "folds").glob("*/train.jsonl"))[0]
    dst_f = dst_root / "derived" / "folds" / src_f.parent.name / "train.jsonl"
    n = same = 0
    mag_err = 0.0
    dir_same = 0
    # keyed by record_id rather than positional, because the excluded-offset filter
    # legitimately removes lines and a positional zip would then compare the wrong pairs
    dst_by_id = {}
    for line in open(dst_f):
        r = json.loads(line)
        dst_by_id[r["record_id"]] = r
    for a in open(src_f):
        ra = json.loads(a)
        rb = dst_by_id.get(ra["record_id"])
        if rb is None:
            continue
        assert ra["quiet"] == rb["quiet"], "quiet flag changed"
        assert ra["observation_refs"] == rb["observation_refs"], "observation changed"
        assert ra["nominal_action"] == rb["nominal_action"], "nominal changed"
        cl = clip_nominal(ra["nominal_action"])
        da = [ra["executed_action"][i] - cl[i] for i in range(3)]
        db = [rb["executed_action"][i] - cl[i] for i in range(3)]
        na = math.sqrt(sum(x * x for x in da))
        nb = math.sqrt(sum(x * x for x in db))
        n += 1
        if na <= 1e-9:
            assert nb <= 1e-9, "a record the shield never touched was modified"
            continue
        if mode == "sft0":
            assert nb <= 1e-9, f"sft0 left a projection in place (norm {nb})"
        else:
            mag_err = max(mag_err, abs(na - nb))
            cos = sum(x * y for x, y in zip(da, db)) / (na * nb)
            if cos > 0.999:
                dir_same += 1
        # unchanged channels
        for i in range(3, 7):
            assert abs(ra["executed_action"][i] - rb["executed_action"][i]) < 1e-12, \
                "sham/sft0 must not touch rotation or gripper channels"
        if ra["executed_action"] == rb["executed_action"]:
            same += 1
    assert same == 0, f"{same} touched records came out unchanged -- transform was a no-op"
    print(f"VERIFY[{mode}] records={n} unchanged_touched={same} "
          f"max_magnitude_error={mag_err:.2e} direction_preserved={dir_same}")
    if mode == "sham":
        assert mag_err < 1e-9, "sham must preserve magnitude exactly"
        # A uniform draw lands within cos>0.999 of the original with probability ~2.5e-4,
        # so a handful of coincidental near-matches in thousands of records is expected
        # and is not evidence the rotation failed. What would be evidence is a
        # substantial fraction, which is what this bound catches.
        assert dir_same <= max(5, 0.01 * n), \
            f"sham reproduced the original direction on {dir_same}/{n} records"
    print(f"ABLATION_ROOT_VERIFIED {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--mode", choices=("sft0", "sham"), required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--exclude-offsets", default="",
                    help="comma-separated offsets to drop from every jsonl")
    a = ap.parse_args()
    src, dst = pathlib.Path(a.src), pathlib.Path(a.dst)
    excl = {x.strip() for x in a.exclude_offsets.split(",") if x.strip()}
    stats = build(src, dst, a.mode, a.seed, excl)
    print(f"BUILT {a.mode} folds={stats['folds']} train_records={stats['records']} "
          f"excluded={stats.get('excluded', 0)} shield_touched={stats['touched']} "
          f"mean_in_norm={stats['norm_sum']/max(stats['touched'],1):.4f} "
          f"mean_out_norm={stats['norm_sum_out']/max(stats['touched'],1):.4f}")
    verify(src, dst, a.mode)


if __name__ == "__main__":
    main()
