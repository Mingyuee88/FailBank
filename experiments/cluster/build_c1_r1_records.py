#!/usr/bin/env python3
"""Build the R1 arms: does staging the failure bank by lead time beat not staging it?

Same dose-matched construction as `build_sec_r0_records.py`. Every arm keeps the same
records, episodes, observations, quiet flags, record count and optimizer steps; the only
thing that varies is which records still carry a projection delta, and whether it is real.

  s1_only   keep the delta on S1_early records (lead > 30 steps before the episode's first
            crossing of cost_pair_min_distance < 0.1087); zero it everywhere else. The
            curriculum's stage 1, and the treatment.
  matched   keep the delta on a random subset of the SAME SIZE drawn WITHIN each teacher
            bucket, so bucket composition is identical to s1_only. Separates "staging by
            lead time helps" from "less teacher signal helps" and from "the buckets that
            early corrections happen to fall in are the useful ones".
  sham      keep the S1_early records but replace each delta with a random direction of the
            SAME magnitude. Separates the CONTENT of early corrections from their dose and
            placement -- the control that killed this project's external-memory line
            (2026-08-11, McNemar p = 1.000), preregistered rather than run afterwards.
  all       keep every delta. This is "use the failure data but do not stage it", i.e. the
            existing recipe applied to the new bank, and the arm the curriculum must beat.
            Its dose is ~9x s1_only's by construction; that is the point of a curriculum,
            not a flaw, but it means s1_only-vs-all is NOT dose-matched and only
            s1_only-vs-matched isolates staging.

Deltas are zeroed rather than records dropped, so state distribution and optimizer steps
stay identical. Rotation and gripper channels are never touched: the CBF projection only
ever writes translation.
"""
from __future__ import annotations
import argparse, json, math, os, pathlib, random, shutil, collections

MAX_TRANSLATION = 1.0
TREATMENT_STAGE = "S1_early"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--dst-root", required=True)
    ap.add_argument("--mode", required=True, choices=["s1_only", "matched", "sham", "all"])
    ap.add_argument("--folds", default="offset_0", help="comma-separated fold names")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()

    src, dst = pathlib.Path(a.src_root), pathlib.Path(a.dst_root)
    want_folds = set(a.folds.split(","))
    fold_dir = src / "derived" / "folds"

    # ---- pass 1: decide which records keep their delta, per fold ----
    # keyed by (fold, record_id): one record appears in several folds and a record_id-keyed
    # dict would let a later fold overwrite an earlier fold's random draw.
    keep = {}
    stage_of = {}
    for fold in sorted(fold_dir.glob("*")):
        if fold.name not in want_folds:
            continue
        f = fold / "train.jsonl"
        if not f.exists():
            continue
        touched = []
        with f.open() as fh:
            for line in fh:
                r = json.loads(line)
                if not (r.get("formal_projection") or {}).get("triggered"):
                    continue
                st = r.get("curriculum_stage")
                stage_of[r["record_id"]] = st
                touched.append((r["record_id"], (r.get("teacher") or {}).get("bucket"), st))
        s1 = [rid for rid, _, st in touched if st == TREATMENT_STAGE]
        if a.mode == "all":
            sel = {rid for rid, _, _ in touched}
        elif a.mode in ("s1_only", "sham"):
            sel = set(s1)
        else:  # matched
            rng = random.Random(a.seed)
            by_b = collections.defaultdict(list)
            for rid, b, _ in touched:
                by_b[b].append(rid)
            need = collections.Counter(b for rid, b, st in touched if st == TREATMENT_STAGE)
            sel = set()
            for b, lst in by_b.items():
                lst = sorted(lst)
                rng.shuffle(lst)
                sel.update(lst[:need[b]])
        for rid, _, _ in touched:
            keep[(fold.name, rid)] = rid in sel
        by_stage = collections.Counter(st for rid, _, st in touched if rid in sel)
        print(f"  {fold.name}: triggered={len(touched)} S1={len(s1)} keeping={len(sel)} "
              f"stages_kept={dict(by_stage)}")

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
        if fold.name not in want_folds:
            continue
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
                        elif a.mode == "sham" and norm > 1e-9:
                            u = random_unit3(rng)
                            new[:3] = [cl[i] + norm * u[i] for i in range(3)]
                            n_shammed += 1
                        else:
                            n_kept += 1
                        r["executed_action"] = new
                    fout.write(json.dumps(r) + "\n")
    print(f"mode={a.mode} kept={n_kept} shammed={n_shammed} zeroed={n_zeroed}")

    # ---- verify: the transform did exactly what it claims ----
    # The specific hazard is emitting a root byte-identical to the source, which would make
    # the arm a silent duplicate and read as a decisive null.
    fname = sorted(want_folds)[0]
    src_f = fold_dir / fname / "train.jsonl"
    dst_f = dst / "derived" / "folds" / fname / "train.jsonl"
    by_id = {}
    with dst_f.open() as fh:
        for line in fh:
            r = json.loads(line)
            by_id[r["record_id"]] = r
    n = kept_v = zero_v = sham_v = 0
    kept_stages = collections.Counter()
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
            na = math.sqrt(sum((ra["executed_action"][i] - cl[i]) ** 2 for i in range(3)))
            nb = math.sqrt(sum((rb["executed_action"][i] - cl[i]) ** 2 for i in range(3)))
            if na <= 1e-9:
                assert nb <= 1e-9, "an untouched record acquired a delta"
                continue
            n += 1
            if not keep.get((fname, ra["record_id"]), False):
                assert nb <= 1e-9, f"a zeroed record kept a delta (norm {nb})"
                zero_v += 1
            elif a.mode == "sham":
                assert abs(na - nb) < 1e-9, f"sham changed magnitude ({na} -> {nb})"
                sham_v += 1
                kept_stages[ra.get("curriculum_stage")] += 1
            else:
                assert abs(na - nb) < 1e-12, "a kept record's delta was altered"
                kept_v += 1
                kept_stages[ra.get("curriculum_stage")] += 1
    print(f"VERIFY ok: touched={n} kept={kept_v} shammed={sham_v} zeroed={zero_v}")
    print(f"  stages actually carrying a delta: {dict(kept_stages)}")
    if a.mode in ("s1_only", "sham"):
        assert set(kept_stages) <= {TREATMENT_STAGE}, \
            f"{a.mode} leaked non-S1 stages into training: {dict(kept_stages)}"
    if a.mode == "all":
        assert len(kept_stages) > 1, "all-arm carries only one stage -- staging leaked in"
    print("C1_R1_BUILD_VERIFIED")


if __name__ == "__main__":
    main()
