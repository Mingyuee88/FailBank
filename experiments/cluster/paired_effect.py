#!/usr/bin/env python3
"""Cell-by-cell: on the exact cells where the layer fired, did it help or hurt?

Aggregate means cannot answer this. The layer only acts on some cells, so a mean over all
100 mixes "cells it changed" with "cells it never touched", and the untouched ones are
identical to the baseline by construction. Pairing on (offset, repeat) and restricting to
fired cells isolates the layer's actual effect.

Every cell where it did not fire has CC exactly 0 in all three tasks, so the trigger is not
misfiring on clean cells -- whatever is going wrong is downstream of the decision to act.
"""
import json, pathlib, collections
import numpy as np

LR = pathlib.Path("${WORK_ROOT}/lora_dagger")

def load(root, arm):
    out = {}
    d = root / arm
    if not d.is_dir(): return out
    for c in sorted(d.glob("off*_r*")):
        f = c / "result.json"
        if not f.is_file(): continue
        try: r = json.load(open(f))
        except Exception: continue
        if r.get("status") != "pass": continue
        md = r.get("metric_decomposition") or {}
        fired = 0
        srd = c / "srd.jsonl"
        if srd.is_file():
            for line in srd.open(errors="ignore"):
                if '"contact_abort"' in line: fired += 1
        out[c.name] = dict(sr=int((r.get("successes") or 0) > 0),
                           cc=float(md.get("policy_induced_cc") or 0), fired=fired)
    return out

for tag, name in (("t1","L1-T1"), ("t2","L1-T2"), ("t4","L1-T4")):
    base = load(LR/("multi_"+tag), "base")
    cbf  = load(LR/("cbfcl_"+tag), "dhead")
    lift = load(LR/("vhead_"+tag), "dhead")
    shared = sorted(set(base) & set(cbf))
    fired = [k for k in shared if cbf[k]["fired"] > 0]
    quiet = [k for k in shared if cbf[k]["fired"] == 0]
    print("=== %s   %d paired cells, layer fired on %d ===" % (name, len(shared), len(fired)))
    for label, keys in (("fired cells", fired), ("untouched cells", quiet)):
        if not keys: continue
        b_cc = np.mean([base[k]["cc"] for k in keys]); c_cc = np.mean([cbf[k]["cc"] for k in keys])
        b_sr = 100*np.mean([base[k]["sr"] for k in keys]); c_sr = 100*np.mean([cbf[k]["sr"] for k in keys])
        worse = sum(1 for k in keys if cbf[k]["cc"] > base[k]["cc"] + 1e-9)
        better = sum(1 for k in keys if cbf[k]["cc"] < base[k]["cc"] - 1e-9)
        print("  %-16s n=%3d   CC %6.1f -> %6.1f   SR %5.1f -> %5.1f   worse %3d / better %3d"
              % (label, len(keys), b_cc, c_cc, b_sr, c_sr, worse, better))
    if fired:
        d = np.array([cbf[k]["cc"] - base[k]["cc"] for k in fired])
        print("  per-cell CC change on fired cells: median %+.1f  p25 %+.1f  p75 %+.1f  max %+.0f"
              % (np.median(d), np.percentile(d,25), np.percentile(d,75), d.max()))
        # is the damage concentrated in a few cells or spread evenly?
        top = np.sort(d)[::-1][:5]
        print("  five worst cells contribute %+.0f of a total change of %+.0f"
              % (top.sum(), d.sum()))
    print()
