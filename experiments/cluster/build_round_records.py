#!/usr/bin/env python3
"""Build one round's training arms for the multi-round self-evolution loop.

THE CORRECTION THIS FIXES. R1 zeroed every non-selected record, i.e. set its target to the
policy's own (clipped) action. On the old success-only bank that was harmless -- there the
nominal IS a good action and zeroing means "keep doing this". On a failure bank it is not:
the nominal on a crash trajectory is the action that caused the crash, so zeroing trains the
policy to repeat it. Measured in R1's own training fold: 389 targets pointing at the policy's
own action on FAILURE trajectories against 225 carrying a real correction -- 1.7 : 1 against.
All three zeroing arms lost 6-10 successes and doubled crash count.

So records are now DROPPED, not zeroed, when their target would be a bad action:

  drop   eventual_success == False and stage != S1_early
         (a failure trajectory's own action is not a teaching target)
  drop   stage in {POST_crossing, S3_emergency}
         (damage handling after contact, and emergency corrections E11 measured at
          1.45-1.54x outside the policy's action support -- unlearnable by imitation)
  keep   everything else. Records from SUCCESSFUL trajectories keep target = nominal and
         act as the anchor/replay set: their nominal is a good action, and they are exactly
         the population whose support was measured regressing when left untrained
         (E13: 18/20 held-out, p = 7.6e-5).

Both arms share ONE record set, so the contrast stays dose-matched in records, states and
optimizer steps; they differ only in which of those records carry a real delta:

  curriculum   delta on S1_early only            -- staged, the treatment
  static       delta on S1_early + S2_mid + NO_RISK -- every non-harmful correction at once,
                                                      i.e. the same data without staging
"""
from __future__ import annotations
import argparse, json, math, os, pathlib, random, shutil, collections

MAX_T = 1.0
DROP_STAGES = {"POST_crossing", "S3_emergency"}
# Curriculum phases. The staged arm trains phase 1 then CONTINUES from its own folded
# checkpoint on phase 2; the unstaged arm gets phase 2's data from the start, for the same
# total number of steps. Same data, same budget, same starting point -- only the ORDER of
# release differs, which is the only construction that actually tests "easy to hard".
DELTA_SETS = {
    "s1":     {"S1_early"},                       # curriculum phase 1
    "s1s2":   {"S1_early", "S2_mid"},             # curriculum phase 2, and the static arm
    "s1s2nr": {"S1_early", "S2_mid", "NO_RISK"},  # the old SE_R1 "static" set, kept for reference
}


def clip_nominal(nom):
    """Clipped translation channels only -- the shield never writes rotation/gripper."""
    return [max(-MAX_T, min(MAX_T, float(x))) for x in nom[:3]]


def keep_record(r):
    """False = this record must not appear in training at all."""
    stage = r.get("curriculum_stage")
    if stage in DROP_STAGES:
        return False
    if not r.get("eventual_success"):
        # From a crash trajectory, ONLY a genuine correction may be kept. Anything else
        # carries target = the policy's own action, i.e. the action that caused the crash.
        # Measured on this fold: 108 such records exist, all untriggered. With
        # quiet_weight = 0.0 they carry no loss weight today, so dropping them costs
        # nothing -- but keeping them would silently become 108 crash demonstrations the
        # moment quiet_weight is raised, which is exactly the class of latent trap this
        # project has been bitten by four times.
        if stage != "S1_early":
            return False
        if not (r.get("formal_projection") or {}).get("triggered"):
            return False
        if math.dist(clip_nominal(r["nominal_action"]), r["executed_action"][:3]) <= 1e-9:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--dst-root", required=True)
    ap.add_argument("--mode", required=True, choices=sorted(DELTA_SETS))
    ap.add_argument("--fold", default="offset_0")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    src, dst = pathlib.Path(a.src_root), pathlib.Path(a.dst_root)
    delta_stages = DELTA_SETS[a.mode]

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

    src_fold = src / "derived" / "folds" / a.fold
    out_fold = dst / "derived" / "folds" / a.fold
    out_fold.mkdir(parents=True)

    stats = collections.Counter()
    delta_by_stage = collections.Counter()
    dropped_by_reason = collections.Counter()
    for f in sorted(src_fold.glob("*")):
        if f.suffix != ".jsonl":
            shutil.copy2(f, out_fold / f.name)
            continue
        with f.open() as fin, (out_fold / f.name).open("w") as fout:
            for line in fin:
                r = json.loads(line)
                if f.name == "train.jsonl":
                    stage = r.get("curriculum_stage")
                    if not keep_record(r):
                        stats["dropped"] += 1
                        if stage in DROP_STAGES:
                            dropped_by_reason[stage] += 1
                        else:
                            dropped_by_reason["failure_traj_no_real_correction"] += 1
                        continue
                    stats["kept"] += 1
                    trig = (r.get("formal_projection") or {}).get("triggered")
                    if trig:
                        cl = clip_nominal(r["nominal_action"])   # 3 values
                        new = list(r["executed_action"])
                        if stage in delta_stages:
                            d = math.dist(cl, r["executed_action"][:3])
                            if d > 1e-9:
                                delta_by_stage[stage] += 1
                                stats["delta_kept"] += 1
                            else:
                                stats["delta_absent"] += 1
                        else:
                            new[:3] = cl[:3]
                            stats["anchored"] += 1
                        r["executed_action"] = new
                fout.write(json.dumps(r) + "\n")

    print(f"mode={a.mode}")
    print(f"  records kept={stats['kept']} dropped={stats['dropped']}  {dict(dropped_by_reason)}")
    print(f"  triggered: real delta={stats['delta_kept']} anchored(target=nominal)={stats['anchored']}")
    print(f"  delta by stage: {dict(delta_by_stage)}")

    # ---- verify ----
    kept_fail_bad = 0
    delta_stage_seen = collections.Counter()
    anchor_from_failure = 0
    with (out_fold / "train.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            stage = r.get("curriculum_stage")
            assert stage not in DROP_STAGES, f"a dropped stage survived: {stage}"
            if not r.get("eventual_success"):
                assert stage == "S1_early", \
                    f"a failure-trajectory record with stage {stage} survived"
                cl = clip_nominal(r["nominal_action"])
                if math.dist(cl, r["executed_action"][:3]) <= 1e-9:
                    anchor_from_failure += 1
            if (r.get("formal_projection") or {}).get("triggered"):
                cl = clip_nominal(r["nominal_action"])
                if math.dist(cl, r["executed_action"][:3]) > 1e-9:
                    delta_stage_seen[stage] += 1
    assert set(delta_stage_seen) <= delta_stages, \
        f"delta leaked into stages outside {delta_stages}: {dict(delta_stage_seen)}"
    # the whole point of the correction: no target may point at a failure trajectory's own action
    assert anchor_from_failure == 0, \
        f"{anchor_from_failure} failure-trajectory records still carry target=nominal"
    print(f"  VERIFY: delta stages={dict(delta_stage_seen)}  "
          f"failure-traj targets pointing at own action={anchor_from_failure}")
    print("ROUND_BUILD_VERIFIED")


if __name__ == "__main__":
    main()
