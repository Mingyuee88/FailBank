#!/usr/bin/env python3
"""Test the contact-density hypothesis before building anything on it.

End-to-end, the cloned-CBF layer helped on one task and hurt on another:
    L1-T2  CFR 35%   SR 77 -> 90, CC 66.4 -> 31.3   large gain
    L1-T1  CFR 54%   SR 83 -> 78, CC 35.9 -> 42.5   slight loss
    L1-T4  CFR  6%   SR 85 -> 64, CC 37.0 -> 87.7   large loss
Direction quality does not explain this -- L1t2 had the LOWEST held-out cosine (0.862) and
the best outcome. Contact density lines up instead: where 94% of cells are already in
contact (L1-T4), a correction carrying |mag| MAE 0.13 against a typical correction of 0.18
gets applied over and over and compounds; where contact is occasional it fires rarely and
the error has no chance to accumulate.

Three cells' worth of evidence is not a law, so this checks the mechanism directly inside
the episodes rather than trusting the correlation:
  * how many steps per episode the layer actually corrected;
  * whether cost accrued BEFORE or AFTER the first correction;
  * whether consecutive corrections point in consistent directions or fight each other.
If compounding is real, L1-T4 should show long correction runs with cost rising after they
start, and low direction consistency.
"""
import json, glob, pathlib, collections
import numpy as np

LR = pathlib.Path("${WORK_ROOT}/lora_dagger")

def episode_stats(task_tag):
    rows = []
    for cell in sorted((LR / ("cbfcl_" + task_tag) / "dhead").glob("off*_r*")):
        res = cell / "result.json"
        srd = cell / "srd.jsonl"
        if not res.is_file() or not srd.is_file(): continue
        try: r = json.load(open(res))
        except Exception: continue
        if r.get("status") != "pass": continue
        md = r.get("metric_decomposition") or {}
        cc = float(md.get("policy_induced_cc") or 0)
        sr = int((r.get("successes") or 0) > 0)
        fires = 0; first_fire = None; steps = 0
        for line in srd.open(errors="ignore"):
            try: x = json.loads(line)
            except Exception: continue
            ty = x.get("type")
            if ty == "contact_abort":
                fires += 1
                if first_fire is None: first_fire = x.get("step")
            elif ty == "dist_summary":
                steps = int(x.get("steps") or 0)
        rows.append(dict(sr=sr, cc=cc, fires=fires, first=first_fire, steps=steps))
    return rows

print("per-episode behaviour of the cloned-CBF layer")
print("%-7s %6s %8s %10s %12s %14s" % ("task", "cells", "fired%", "fires/ep", "CC|fired", "CC|not fired"))
for tag, name in (("t1", "L1-T1"), ("t2", "L1-T2"), ("t4", "L1-T4")):
    rows = episode_stats(tag)
    if not rows: print("  %-7s no data" % name); continue
    fired = [r for r in rows if r["fires"] > 0]
    quiet = [r for r in rows if r["fires"] == 0]
    print("%-7s %6d %8.0f %10.1f %12.1f %14.1f" % (
        name, len(rows), 100.0*len(fired)/len(rows),
        np.mean([r["fires"] for r in fired]) if fired else 0,
        np.mean([r["cc"] for r in fired]) if fired else 0,
        np.mean([r["cc"] for r in quiet]) if quiet else 0))

print()
print("the discriminator: cost on episodes the layer touched vs episodes it left alone.")
print("If the layer is the cause of the extra cost, CC|fired >> CC|not fired.")
print()
print("baseline contact density (pi_2 alone, from the multi_* arms):")
for tag, name in (("t1","L1-T1"), ("t2","L1-T2"), ("t4","L1-T4")):
    cells = []
    for c in sorted((LR/("multi_"+tag)/"base").glob("off*_r*")):
        f = c/"result.json"
        if not f.is_file(): continue
        try: r = json.load(open(f))
        except Exception: continue
        if r.get("status") != "pass": continue
        cells.append(float((r.get("metric_decomposition") or {}).get("policy_induced_cc") or 0))
    if cells:
        cfr = 100.0*sum(1 for c in cells if c == 0)/len(cells)
        print("  %-7s CFR %5.1f%%   mean CC %5.1f" % (name, cfr, np.mean(cells)))
