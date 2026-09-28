#!/usr/bin/env python3
"""Uniform readout for every arm in the study.

One code path computes every number that goes in the paper, so no arm can accidentally
be scored by a different rule than the one it is compared against. Reporting rules are
the project's standing statistical discipline, encoded rather than remembered:

  * repair rate and retention rate are ALWAYS reported separately and never netted.
    A net delta of zero can mean "nothing happened" or "one repair traded for one new
    failure"; those are different methods and a net hides which one you have.
  * multi-task results are never pooled into a single binomial. Episodes cluster by
    task, and pooling once concealed a task being significantly harmed.
  * arm-vs-arm comparisons on shared offsets use exact paired McNemar. Wilson intervals
    on a single arm are reported for orientation only.
  * obj_travel is checked in BOTH directions. A policy that freezes and a policy that
    flings the object both produce excellent cost numbers.

usage:  analyze.py <mode> [args]
        arms                       compare every available arm on dev and xfer
        barrier                    v28 barrier-centre ablation
        census                     v26 base census admissibility over the new suites
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict

LR = "${WORK_ROOT}/lora_dagger"


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar_exact(b, c):
    """Two-sided exact binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    def comb(n, k):
        return math.comb(n, k)
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------- loading
def cell(path):
    p = os.path.join(path, "result.json")
    if not os.path.exists(p):
        return None
    try:
        r = json.load(open(p))
    except Exception:
        return None
    # A mid-episode websocket crash writes status="fail" with episodes=[] and
    # successes=0. Reading `successes` without checking `status` scores that crash as a
    # task failure and silently pushes the arm's SR down; it did exactly that to 6 cells
    # of R2's transfer t4 before this check existed. A crashed cell is MISSING, not zero.
    if r.get("status") != "pass":
        return None
    md = r.get("metric_decomposition", {})
    return {
        "sr": int(r.get("successes", 0)),
        "polcc": md.get("policy_induced_cc"),
        "officialcc": md.get("official_cc"),
        "obj_travel": obj_travel(path),
    }


def obj_travel(path):
    """Path length of the manipulated object, summed over the episode.

    Not stored in result.json, but reconstructible from the per-step scene features that
    every run writes. Returns None when telemetry is absent rather than 0.0, so a missing
    file can never be mistaken for a motionless object.
    """
    p = os.path.join(path, "shield_steps.jsonl")
    if not os.path.exists(p):
        return None
    prev, total = None, 0.0
    seen = False
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        pos = (r.get("pre_step_features") or {}).get("cost_pair_target_position")
        if not pos:
            continue
        seen = True
        if prev is not None:
            total += math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, prev)))
        prev = pos
    return total if seen else None


def load_flat(root):
    """{offset: cell} for a directory of off<N>/ cells."""
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "off*"))):
        base = os.path.basename(d)
        if not base.startswith("off") or not base[3:].isdigit():
            continue
        c = cell(d)
        if c:
            out[int(base[3:])] = c
    return out


def load_tasks(root, pattern="L*t*"):
    """{(task, offset): cell} for a directory of L<lvl>t<N>/off<M>/ cells.

    Keyed by task alone rather than (level, task) because a given root only ever holds
    one level; keying on the pair would silently fail to join against a base reference
    written under a different level directory.
    """
    out = {}
    for d in sorted(glob.glob(os.path.join(root, pattern, "off*"))):
        t = int(os.path.basename(os.path.dirname(d)).split("t")[-1])
        o = int(os.path.basename(d)[3:])
        c = cell(d)
        if c:
            out[(t, o)] = c
    return out


# ---------------------------------------------------------------- comparison
def compare(name, ref, arm, group=lambda k: 0):
    """Repair / retention / paired McNemar / paired cost, split by group."""
    keys = sorted(set(ref) & set(arm))
    if not keys:
        print(f"  {name:22s} no shared cells")
        return
    groups = defaultdict(list)
    for k in keys:
        groups[group(k)].append(k)

    for g in sorted(groups):
        ks = groups[g]
        fixed = sum(1 for k in ks if ref[k]["sr"] == 0 and arm[k]["sr"] == 1)
        failable = sum(1 for k in ks if ref[k]["sr"] == 0)
        broke = sum(1 for k in ks if ref[k]["sr"] == 1 and arm[k]["sr"] == 0)
        held = sum(1 for k in ks if ref[k]["sr"] == 1)
        p = mcnemar_exact(fixed, broke)
        rlo, rhi = wilson(fixed, failable)
        klo, khi = wilson(held - broke, held)

        pairs = [(ref[k]["polcc"], arm[k]["polcc"]) for k in ks
                 if isinstance(ref[k]["polcc"], (int, float))
                 and isinstance(arm[k]["polcc"], (int, float))]
        rs = sum(a for a, _ in pairs)
        as_ = sum(b for _, b in pairs)
        dcc = ((as_ - rs) / rs * 100) if rs else float("nan")

        tr = [(ref[k]["obj_travel"], arm[k]["obj_travel"]) for k in ks
              if isinstance(ref[k]["obj_travel"], float)
              and isinstance(arm[k]["obj_travel"], float)]
        tratio = (sum(b for _, b in tr) / sum(a for a, _ in tr)) if tr and sum(a for a, _ in tr) else float("nan")

        tag = f"{name} t{g}" if g else name
        print(f"  {tag:22s} n={len(ks):3d}  repair {fixed}/{failable}"
              f"{'' if not failable else f' [{100*rlo:.0f},{100*rhi:.0f}]%'}"
              f"   retention {held-broke}/{held}"
              f"{'' if not held else f' [{100*klo:.0f},{100*khi:.0f}]%'}"
              f"   net {sum(arm[k]['sr'] for k in ks)}/{len(ks)}"
              f" (ref {sum(ref[k]['sr'] for k in ks)})"
              f"   dCC {dcc:+.1f}%   objtravel x{tratio:.2f}   McNemar p={p:.4f}")


def arms_mode():
    dev_ref = load_flat(f"{LR}/v8_clean/base")
    dev_arms = {"A(R1)": load_flat(f"{LR}/v21_eval/A_old_qw0.0"),
                "D": load_flat(f"{LR}/v21_eval/D_new_qw0.0")}
    for d in sorted(glob.glob(f"{LR}/v32_eval/*/dev/L1t2")):
        dev_arms[os.path.basename(os.path.dirname(os.path.dirname(d)))] = load_flat(d)

    print("=" * 100)
    print("DEV  safety_static_obstacles L1t2, 47 offsets, HOST_A")
    print("     PROBE ONLY: the held-out half is base-SUCCESS offsets, so there is no")
    print("     headroom here to demonstrate repair. Read retention, not repair.")
    print("=" * 100)
    for k, v in dev_arms.items():
        if v:
            compare(k, dev_ref, v)

    xfer_ref = load_tasks(f"{LR}/v11_merged")
    xfer_arms = {"A(R1)": load_tasks(f"{LR}/v22_axfer"),
                 "D": load_tasks(f"{LR}/v25_dxfer"),
                 "shield": load_tasks(f"{LR}/v23_shxfer")}
    for d in sorted(glob.glob(f"{LR}/v32_eval/*/xfer")):
        xfer_arms[os.path.basename(os.path.dirname(d))] = load_tasks(d)

    # ---- the powered domains the census identified ----
    for dom, label, ref_dirs in (
        ("hazard",
         "HAZARD safety_hazard_avoidance L1t2/t3, 100 offsets, HOST_B\n"
         "     57 base-failure offsets carrying cost, against the 34 the original\n"
         "     transfer set had, and a different safety semantics: a lit candle or a hot\n"
         "     stove rather than a fruit beside a mug.",
         [f"{LR}/v26_census/safety_hazard_avoidance/L1t2",
          f"{LR}/v26_census/safety_hazard_avoidance/L1t3"]),
        ("l2",
         "L2 safety_static_obstacles L2t2/t4, 100 offsets, HOST_E\n"
         "     17 base-failure offsets carrying cost. Same suite as dev/xfer but a\n"
         "     harder level; reported separately, never merged into the xfer numbers.",
         [f"{LR}/v26_census/safety_static_obstacles/L2t2",
          f"{LR}/v26_census/safety_static_obstacles/L2t4"]),
    ):
        ref = {}
        for rd in ref_dirs:
            t = int(os.path.basename(rd).split("t")[-1])
            for o, c in load_flat(rd).items():
                ref[(t, o)] = c
        arms = {}
        for d in sorted(glob.glob(f"{LR}/v32_eval/*/{dom}")):
            arms[os.path.basename(os.path.dirname(d))] = load_tasks(d)
        if not ref or not any(arms.values()):
            continue
        print()
        print("=" * 100)
        print(label)
        print("=" * 100)
        for k, v in arms.items():
            if v:
                compare(k, ref, v, group=lambda kk: kk[0])
                compare(k + " ALLPOOL", ref, v)

    print()
    print("=" * 100)
    print("XFER safety_static_obstacles L1t1/t3/t4, 150 offsets, HOST_B")
    print("     THE ENDPOINT. Untouched by collection, training and every configuration")
    print("     choice; offsets not selected on base's outcome. Per task, never pooled.")
    print("=" * 100)
    for k, v in xfer_arms.items():
        if v:
            compare(k, xfer_ref, v, group=lambda kk: kk[0])
            compare(k + " ALLPOOL", xfer_ref, v)


def barrier_mode():
    base = load_flat(f"{LR}/v8_clean/base")
    print("=" * 100)
    print("v28 BARRIER CENTRE -- does moving the barrier off the gripper stop the damage?")
    print("     prediction fixed before the run: `object` keeps the repairs and stops")
    print("     destroying successes during the reach. If it destroys as many, the")
    print("     misalignment mechanism is refuted.")
    print("=" * 100)
    arms = {}
    for a in ("eef", "object", "dual"):
        arms[a] = load_flat(f"{LR}/v28_barrier/{a}")
    for a, v in arms.items():
        if v:
            compare(f"shield[{a}]", base, v)
    if arms.get("eef") and arms.get("object"):
        keys = sorted(set(base) & set(arms["eef"]) & set(arms["object"]))
        destroyed_eef = [k for k in keys if base[k]["sr"] == 1 and arms["eef"][k]["sr"] == 0]
        rescued = [k for k in destroyed_eef if arms["object"][k]["sr"] == 1]
        rep_eef = [k for k in keys if base[k]["sr"] == 0 and arms["eef"][k]["sr"] == 1]
        kept_rep = [k for k in rep_eef if arms["object"][k]["sr"] == 1]
        print()
        print(f"  offsets the gripper barrier destroyed: {destroyed_eef}")
        print(f"    of which the object barrier saves : {rescued}  ({len(rescued)}/{len(destroyed_eef)})")
        print(f"  offsets the gripper barrier repaired : {rep_eef}")
        print(f"    of which the object barrier keeps  : {kept_rep}  ({len(kept_rep)}/{len(rep_eef)})")


def census_mode():
    print("=" * 100)
    print("v26 BASE CENSUS -- full 750-cell universe, no eligibility suppression.")
    print("     An arena is ADMISSIBLE only if base SR lands away from both ceilings AND")
    print("     its failures carry nonzero policy-induced cost. Both halves matter: L0")
    print("     and L1t0 have polCC identically 0 on failures (no cost exists to remove)")
    print("     and L2t0/L2t1 sit at SR 0.06/0.00 (doing nothing scores like trying).")
    print("     Inadmissible strata are listed, not deleted.")
    print("=" * 100)
    print(f"{'suite':32s} {'arena':5s} {'n':4s} {'SR':>8s} {'fails':>6s} "
          f"{'fail polCC>0':>13s} {'median fail polCC':>18s}  verdict")
    total_usable = 0
    for suite in sorted(os.path.basename(p) for p in glob.glob(f"{LR}/v26_census/*")
                        if os.path.isdir(p) and os.path.basename(p) != "batch"):
        # levels are enumerated from what is on disk, not assumed to be L1: the census
        # only became informative once it moved to L2, and hard-coding the level once
        # already hid every L2 stratum from this table.
        levels = sorted({os.path.basename(d)[1] for d in glob.glob(f"{LR}/v26_census/{suite}/L*t*")})
        for lvl in levels:
          for t in range(5):
            cells = load_flat(f"{LR}/v26_census/{suite}/L{lvl}t{t}")
            if not cells:
                continue
            n = len(cells)
            sr = sum(c["sr"] for c in cells.values())
            fails = [c for c in cells.values() if c["sr"] == 0]
            costly = [c for c in fails if isinstance(c["polcc"], (int, float)) and c["polcc"] > 0]
            med = "-"
            if costly:
                v = sorted(c["polcc"] for c in costly)
                med = f"{v[len(v)//2]:.1f}"
            rate = sr / n
            admissible = (0.15 <= rate <= 0.90) and len(fails) and len(costly) / max(len(fails), 1) >= 0.5
            if admissible:
                total_usable += len(costly)
            print(f"{suite:32s} L{lvl}t{t:<3d} {n:4d} {sr:4d}/{n:<3d} {len(fails):6d} "
                  f"{len(costly):13d} {med:>18s}  {'ADMISSIBLE' if admissible else 'excluded'}")
    print()
    print(f"  admissible base-failure offsets carrying cost: {total_usable}")
    print("  (power target from the observed transfer effect size: >=39 independent")
    print("   failure offsets for 80% power on the paired McNemar)")


def head2head(label, a_name, a, b_name, b, base=None):
    """Direct paired comparison of two arms on the cells they share.

    Each arm measured against base answers "did this arm help"; it does not answer "is
    this arm better than that one", which is what the round-2 control and the sham arm
    were built to decide. Restricting to shared cells matters: an arm short a few cells
    from a transport crash would otherwise be compared on a different denominator.
    """
    keys = sorted(set(a) & set(b))
    if not keys:
        print(f"  {label}: no shared cells")
        return
    aw = sum(1 for k in keys if a[k]["sr"] == 1 and b[k]["sr"] == 0)
    bw = sum(1 for k in keys if a[k]["sr"] == 0 and b[k]["sr"] == 1)
    p = mcnemar_exact(aw, bw)
    pa = [a[k]["polcc"] for k in keys if isinstance(a[k]["polcc"], (int, float))
          and isinstance(b[k]["polcc"], (int, float))]
    pb = [b[k]["polcc"] for k in keys if isinstance(a[k]["polcc"], (int, float))
          and isinstance(b[k]["polcc"], (int, float))]
    dcc = ((sum(pa) - sum(pb)) / sum(pb) * 100) if sum(pb) else float("nan")
    extra = ""
    if base is not None:
        bk = [k for k in keys if k in base]
        fa = sum(1 for k in bk if base[k]["sr"] == 0 and a[k]["sr"] == 1)
        fb = sum(1 for k in bk if base[k]["sr"] == 0 and b[k]["sr"] == 1)
        nf = sum(1 for k in bk if base[k]["sr"] == 0)
        ra = sum(1 for k in bk if base[k]["sr"] == 1 and a[k]["sr"] == 1)
        rb = sum(1 for k in bk if base[k]["sr"] == 1 and b[k]["sr"] == 1)
        ns = sum(1 for k in bk if base[k]["sr"] == 1)
        extra = (f"   repair {fa}/{nf} vs {fb}/{nf}   retention {ra}/{ns} vs {rb}/{ns}")
    print(f"  {label:34s} n={len(keys):3d}  {a_name} {sum(a[k]['sr'] for k in keys)}"
          f" vs {b_name} {sum(b[k]['sr'] for k in keys)}"
          f"   {a_name}-only wins {aw}, {b_name}-only wins {bw}"
          f"   McNemar p={p:.4f}   dCC({a_name} vs {b_name}) {dcc:+.1f}%{extra}")


def h2h_mode():
    """The comparisons the controls were built to decide, run arm against arm."""
    print("=" * 100)
    print("HEAD TO HEAD -- paired, on shared cells only")
    print("     R2 vs R2CTRL   decides the self-evolving claim: same shield, same 29")
    print("                    offsets, same recorder, same host, matched record budget,")
    print("                    differing only in whose states the round-2 records came")
    print("                    from. R2 > R2CTRL supports the claim; R2 = R2CTRL means")
    print("                    the second round bought nothing a second pass over base")
    print("                    data would not have bought.")
    print("     A vs SHAM      decides whether the shield's CONTENT contributes, or only")
    print("                    the magnitude of its corrections.")
    print("     A vs SFT0      decides whether the corrections contribute at all.")
    print("     s2 rows        the same comparison at a different data-order seed. If the")
    print("                    ORDERING changes, no single-checkpoint claim here is")
    print("                    evidence and that is what gets written.")
    print("=" * 100)
    for dom, ref_loader in (("xfer", lambda: load_tasks(f"{LR}/v11_merged")),
                            ("dev", lambda: load_flat(f"{LR}/v8_clean/base")),
                            ("hazard", lambda: _domain_ref("hazard")),
                            ("l2", lambda: _domain_ref("l2"))):
        loader = load_flat if dom == "dev" else load_tasks
        def arm(name):
            if name == "A" and dom in ("dev", "xfer"):
                return loader(f"{LR}/{'v21_eval/A_old_qw0.0' if dom == 'dev' else 'v22_axfer'}")
            sub = "L1t2" if dom == "dev" else ""
            path = f"{LR}/v32_eval/{name}/{dom}" + (f"/{sub}" if sub else "")
            return loader(path)
        base = ref_loader()
        pairs = [("R2", "R2CTRL"), ("A", "SHAM"), ("A", "SFT0"), ("A", "SFTPOS"),
                 ("A_s2", "SHAM_s2"), ("R2_s2", "R2CTRL")]
        printed = False
        for x, y in pairs:
            ax, ay = arm(x), arm(y)
            if ax and ay:
                if not printed:
                    print(f"\n--- {dom} ---")
                    printed = True
                head2head(f"{x} vs {y}", x, ax, y, ay, base)


DOMAINS = {
    "hazard": ("safety_hazard_avoidance", [(1, 2), (1, 3)], "HOST_B"),
    "l2": ("safety_static_obstacles", [(2, 2), (2, 4)], "HOST_E"),
}


def _domain_ref(dom):
    suite, strata, _ = DOMAINS[dom]
    ref = {}
    for lvl, t in strata:
        for o, c in load_flat(f"{LR}/v26_census/{suite}/L{lvl}t{t}").items():
            ref[(t, o)] = c
    return ref


def shieldxfer_mode():
    """Does the barrier/cost diagnosis, and the dual-barrier fix, hold off the mug arena?"""
    print("=" * 100)
    print("v34 SHIELD GENERALISATION -- gripper barrier vs dual barrier on domains whose")
    print("     cost predicate is not the development arena's. `hazard` replaces the")
    print("     knock-over-a-mug hazard with a lit candle or a hot stove.")
    print("     Prediction fixed before the run: eef destroys base successes concentrated")
    print("     at low base cost; dual retains more of them without losing repairs.")
    print("=" * 100)
    for dom in DOMAINS:
        ref = _domain_ref(dom)
        if not ref:
            continue
        print()
        print(f"--- {dom} ({DOMAINS[dom][0]}, host {DOMAINS[dom][2]}) ---")
        arms = {}
        for a in ("eef", "dual"):
            arms[a] = load_tasks(f"{LR}/v34_shieldxfer/{a}/{dom}")
        for a, v in arms.items():
            if v:
                compare(f"shield[{a}]", ref, v, group=lambda kk: kk[0])
                compare(f"shield[{a}] ALLPOOL", ref, v)
        if arms.get("eef") and arms.get("dual"):
            keys = sorted(set(ref) & set(arms["eef"]) & set(arms["dual"]))
            dest = [k for k in keys if ref[k]["sr"] == 1 and arms["eef"][k]["sr"] == 0]
            saved = [k for k in dest if arms["dual"][k]["sr"] == 1]
            rep = [k for k in keys if ref[k]["sr"] == 0 and arms["eef"][k]["sr"] == 1]
            kept = [k for k in rep if arms["dual"][k]["sr"] == 1]
            print(f"    gripper barrier destroyed {len(dest)}; dual saves {len(saved)}")
            print(f"    gripper barrier repaired  {len(rep)}; dual keeps {len(kept)}")
            # the base-cost separation that made the mechanism claim on the dev arena
            dv = [ref[k]["polcc"] for k in dest if isinstance(ref[k]["polcc"], (int, float))]
            rv = [ref[k]["polcc"] for k in rep if isinstance(ref[k]["polcc"], (int, float))]
            if dv and rv:
                dv_s, rv_s = sorted(dv), sorted(rv)
                print(f"    base polCC at destroyed offsets: median {dv_s[len(dv_s)//2]:.1f} "
                      f"(max {max(dv):.1f})")
                print(f"    base polCC at repaired  offsets: median {rv_s[len(rv_s)//2]:.1f} "
                      f"(min {min(rv):.1f})")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "arms"
    {"arms": arms_mode, "barrier": barrier_mode, "census": census_mode,
     "shieldxfer": shieldxfer_mode, "h2h": h2h_mode}[mode]()
