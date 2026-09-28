#!/usr/bin/env python3
"""Build the R0 arms for the Support-gated Evolving Curriculum.

The question R0 answers: D was trained on ALL 866 outcome-filtered shield corrections,
of which only 484 (55.9%) lie inside its own action support. Does training ONLY on the
in-support subset expand support where training on everything did not?

Arms. Every arm keeps the same records, episodes, observations, quiet flags, record count
and optimizer steps. The ONLY thing that varies is which records still carry a projection
delta, and whether that delta is real:

  gated        keep the delta on S1 records (d_executed_min <= spread_mean); zero it
               elsewhere. The treatment.
  matched      keep the delta on a random subset of the SAME SIZE, drawn WITHIN each
               teacher bucket so the bucket composition is identical to `gated`; zero it
               elsewhere. Separates "support-gating helps" from "less teacher signal helps"
               and from "the buckets gating happens to prefer are the useful ones".
  sham_gated   keep the S1 records, but replace each delta with a random direction of the
               SAME magnitude. Separates the CONTENT of the in-support corrections from
               their dose and placement. This is the control that killed the external-memory
               line of this project (2026-08-11, McNemar p = 1.000); it is preregistered
               here rather than run afterwards.

Deltas are zeroed rather than records dropped, so record count, state distribution and
optimizer steps are identical across arms by construction. Rotation and gripper channels
are never touched, because the CBF projection only ever writes translation.

Blobs and episodes are symlinked, never copied.
"""
from __future__ import annotations
import argparse, json, math, os, pathlib, random, shutil, collections

MAX_TRANSLATION = 1.0


def clip_nominal(nominal):
    out = list(nominal)
    for i in range(3):
        out[i] = max(-MAX_TRANSLATION, min(MAX_TRANSLATION, float(out[i])))
    return out


def random_unit3(rng):
    while True:
        v = [rng.gauss(0.0, 1.0) for _ in range(3)]
        n = math.sqrt(sum(x * x for x in v))
        if n > 1e-9:
            return [x / n for x in v]


def load_support(path):
    """record_id -> (stage, ratio) from an offline support annotation.

    The ratio d_executed_min / spread_mean is the continuous learnability axis; the stage
    is its coarsening. `anti_gated` needs the continuous form to rank within a bucket.
    """
    st, ratio = {}, {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            sp = r.get("spread_mean") or 0.0
            if sp <= 0:
                continue
            x = r["d_executed_min"] / sp
            ratio[r["record_id"]] = x
            st[r["record_id"]] = "S1" if x <= 1.0 else ("S2" if x <= 2.0 else "S3")
    return st, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--dst-root", required=True)
    ap.add_argument("--support", required=True, help="support.jsonl annotated under the CURRENT policy")
    ap.add_argument("--mode", required=True,
                    choices=["gated", "matched", "sham_gated", "anti_gated"])
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()

    src, dst = pathlib.Path(a.src_root), pathlib.Path(a.dst_root)
    stages, ratios = load_support(a.support)
    print(f"support annotation: {len(stages)} records staged")

    # ---- pass 1: decide, per fold, which records keep their delta ----
    fold_dir = src / "derived" / "folds"
    # keyed by (fold, record_id): the same record appears in several folds, and a
    # record_id-keyed dict would let a later fold overwrite an earlier fold's random
    # draw, silently breaking the dose match (observed: 450 kept instead of 484).
    keep = {}          # (fold_name, record_id) -> bool
    for fold in sorted(fold_dir.glob("*")):
        f = fold / "train.jsonl"
        if not f.exists():
            continue
        touched = []   # (record_id, bucket)
        with f.open() as fh:
            for line in fh:
                r = json.loads(line)
                if not (r.get("formal_projection") or {}).get("triggered"):
                    continue
                touched.append((r["record_id"], (r.get("teacher") or {}).get("bucket")))
        s1 = [rid for rid, _ in touched if stages.get(rid) == "S1"]
        if a.mode in ("gated", "sham_gated"):
            sel = set(s1)
        elif a.mode == "anti_gated":
            # Same count WITHIN each bucket as `gated`, but taking the records FURTHEST
            # outside support instead of the closest. Bucket composition is therefore
            # identical to gated/matched, so the contrast isolates the support axis alone
            # and cannot be explained by the outcome filter's own preferences -- which it
            # would be if we simply ranked all 866 by support distance, since `low` is 72%
            # S3 and `high` only 23%.
            by_b = collections.defaultdict(list)
            for rid, b in touched:
                by_b[b].append(rid)
            need = collections.Counter(b for rid, b in touched if stages.get(rid) == "S1")
            sel = set()
            for b, lst in by_b.items():
                lst = sorted(lst, key=lambda r: (-ratios.get(r, 0.0), r))
                sel.update(lst[:need[b]])
        else:  # matched: same count WITHIN each bucket, chosen at random
            rng = random.Random(a.seed)
            by_b = collections.defaultdict(list)
            for rid, b in touched:
                by_b[b].append(rid)
            need = collections.Counter(b for rid, b in touched if stages.get(rid) == "S1")
            sel = set()
            for b, lst in by_b.items():
                lst = sorted(lst)
                rng.shuffle(lst)
                sel.update(lst[:need[b]])
        for rid, _ in touched:
            keep[(fold.name, rid)] = rid in sel
        print(f"  {fold.name}: triggered={len(touched)} S1={len(s1)} keeping={len(sel)}")

    # ---- pass 2: write ----
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for shared in ("blobs", "episodes"):
        s = src / shared
        if s.exists():
            os.symlink(s, dst / shared)
    (dst / "derived").mkdir()
    for extra in (src / "derived").glob("*"):
        if extra.is_file():
            shutil.copy2(extra, dst / "derived" / extra.name)
    (dst / "derived" / "folds").mkdir()

    rng = random.Random(a.seed + 1)
    n_kept = n_zeroed = n_shammed = 0
    for fold in sorted(fold_dir.glob("*")):
        out_fold = dst / "derived" / "folds" / fold.name
        out_fold.mkdir()
        for f in fold.glob("*"):
            if f.suffix != ".jsonl":
                shutil.copy2(f, out_fold / f.name)
                continue
            with f.open() as fin, (out_fold / f.name).open("w") as fout:
                for line in fin:
                    r = json.loads(line)
                    trig = (r.get("formal_projection") or {}).get("triggered")
                    if trig and f.name == "train.jsonl":
                        cl = clip_nominal(r["nominal_action"])
                        d = [float(r["executed_action"][i]) - cl[i] for i in range(3)]
                        norm = math.sqrt(sum(x * x for x in d))
                        new = list(r["executed_action"])
                        if not keep.get((fold.name, r["record_id"]), False):
                            new[:3] = cl[:3]
                            n_zeroed += 1
                        elif a.mode == "sham_gated" and norm > 1e-9:
                            u = random_unit3(rng)
                            new[:3] = [cl[i] + norm * u[i] for i in range(3)]
                            n_shammed += 1
                        else:
                            n_kept += 1
                        r["executed_action"] = new
                    fout.write(json.dumps(r) + "\n")
    print(f"mode={a.mode} kept={n_kept} shammed={n_shammed} zeroed={n_zeroed}")

    # ---- verify: the transform must have done exactly what it claims ----
    # Four silent no-ops in this project produced healthy-looking artifacts. The specific
    # hazard here is emitting a root byte-identical to the source, which would make the
    # arm a duplicate of D and read as a decisive null.
    src_f = sorted(fold_dir.glob("*/train.jsonl"))[0]
    dst_f = dst / "derived" / "folds" / src_f.parent.name / "train.jsonl"
    by_id = {}
    with dst_f.open() as fh:
        for line in fh:
            r = json.loads(line)
            by_id[r["record_id"]] = r
    n = kept_v = zero_v = sham_v = 0
    with src_f.open() as fh:
        for line in fh:
            ra = json.loads(line)
            rb = by_id.get(ra["record_id"])
            assert rb is not None, "record vanished"
            assert ra["observation_refs"] == rb["observation_refs"], "observation changed"
            assert ra["nominal_action"] == rb["nominal_action"], "nominal changed"
            assert ra["quiet"] == rb["quiet"], "quiet changed"
            for i in range(3, 7):
                assert abs(ra["executed_action"][i] - rb["executed_action"][i]) < 1e-12, \
                    "a channel the shield never writes was modified"
            cl = clip_nominal(ra["nominal_action"])
            da = [ra["executed_action"][i] - cl[i] for i in range(3)]
            db = [rb["executed_action"][i] - cl[i] for i in range(3)]
            na = math.sqrt(sum(x * x for x in da))
            nb = math.sqrt(sum(x * x for x in db))
            if na <= 1e-9:
                assert nb <= 1e-9, "an untouched record acquired a delta"
                continue
            n += 1
            if not keep.get((src_f.parent.name, ra["record_id"]), False):
                assert nb <= 1e-9, f"a zeroed record kept a delta (norm {nb})"
                zero_v += 1
            elif a.mode == "sham_gated":
                assert abs(na - nb) < 1e-9, f"sham changed magnitude ({na} -> {nb})"
                cos = sum(x * y for x, y in zip(da, db)) / (na * nb)
                assert cos < 0.999, "sham produced the original direction"
                sham_v += 1
            else:
                assert abs(na - nb) < 1e-12, "a kept record's delta was altered"
                for i in range(3):
                    assert abs(da[i] - db[i]) < 1e-12, "a kept record's direction changed"
                kept_v += 1
    assert n > 0, "no touched records seen -- the arm is a duplicate of the source"
    assert zero_v > 0, "nothing was zeroed -- the gate did nothing"
    if a.mode in ("matched", "anti_gated"):
        print(f"{a.mode} arm dose: kept={kept_v} (must equal the gated arm's S1 count)")
    if a.mode == "anti_gated":
        # Effective-value check: the arm claims to select the records FURTHEST outside
        # support. Assert it, rather than trusting that the sort key had the sign intended.
        kept_r = [ratios[rid] for (fold, rid), v in keep.items()
                  if v and fold == src_f.parent.name and rid in ratios]
        drop_r = [ratios[rid] for (fold, rid), v in keep.items()
                  if not v and fold == src_f.parent.name and rid in ratios]
        assert kept_r and drop_r, "anti_gated: nothing to compare"
        mk = sorted(kept_r)[len(kept_r) // 2]
        md = sorted(drop_r)[len(drop_r) // 2]
        print(f"anti_gated support ratio: kept median {mk:.3f} vs dropped median {md:.3f}")
        assert mk > md, ("anti_gated selected the records CLOSER to support -- the sort "
                         "sign is inverted and this arm is a duplicate of gated")
    print(f"VERIFY ok: touched={n} kept={kept_v} shammed={sham_v} zeroed={zero_v}")
    print("SEC_R0_BUILD_VERIFIED")


if __name__ == "__main__":
    main()
