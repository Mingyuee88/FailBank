#!/usr/bin/env python3
"""Paired effect of the safety layer, against the CORRECT baseline.

Earlier I compared the layer against multi_*/base, which unsets the checkpoint and is
therefore the pi0.5-finetuned BASE model, not pi_2. The layer arms all load
se_curr2/CURRICULUM_p2. So the right baseline is the r2 arm, which is the same policy
without the layer -- the only difference then is the layer itself.
"""
import json, pathlib
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
        fired = 0
        srd = c / "srd.jsonl"
        if srd.is_file():
            for line in srd.open(errors="ignore"):
                if '"contact_abort"' in line: fired += 1
        out[c.name] = dict(sr=int((r.get("successes") or 0) > 0),
                           cc=float((r.get("metric_decomposition") or {}).get("policy_induced_cc") or 0),
                           fired=fired)
    return out

print("baseline = r2 arm (pi_2, same checkpoint, no safety layer)")
print()
for tag, name in (("t1","L1-T1"), ("t2","L1-T2"), ("t4","L1-T4")):
    r2   = load(LR/("multi_"+tag), "r2")
    cbf  = load(LR/("cbfcl_"+tag), "dhead")
    lift = load(LR/("vhead_"+tag), "dhead")
    if not r2 or not cbf:
        print("=== %s incomplete ===" % name); continue
    for lbl, arm in (("cloned CBF", cbf), ("fixed lift", lift)):
        shared = sorted(set(r2) & set(arm))
        if not shared: continue
        fired = [k for k in shared if arm[k]["fired"] > 0]
        b_cc = np.mean([r2[k]["cc"] for k in shared]); a_cc = np.mean([arm[k]["cc"] for k in shared])
        b_sr = 100*np.mean([r2[k]["sr"] for k in shared]); a_sr = 100*np.mean([arm[k]["sr"] for k in shared])
        print("%-7s %-12s all %3d cells   SR %5.1f -> %5.1f   CC %6.1f -> %6.1f"
              % (name, lbl, len(shared), b_sr, a_sr, b_cc, a_cc))
        if fired:
            d = np.array([arm[k]["cc"] - r2[k]["cc"] for k in fired])
            ds = np.array([arm[k]["sr"] - r2[k]["sr"] for k in fired])
            print("        on the %3d cells it fired: CC %+.1f mean, %+.1f median   SR %+.0f%%   worse %d / better %d"
                  % (len(fired), d.mean(), np.median(d), 100*ds.mean(),
                     int((d > 1e-9).sum()), int((d < -1e-9).sum())))
    print()
print("reference, same protocol:  AEGIS and the base model")
for tag, name in (("t1","L1-T1"), ("t2","L1-T2"), ("t4","L1-T4")):
    for arm in ("base", "aegis"):
        a = load(LR/("multi_"+tag), arm)
        if a:
            print("  %-7s %-6s SR %5.1f  CC %6.1f" % (name, arm,
                  100*np.mean([v["sr"] for v in a.values()]), np.mean([v["cc"] for v in a.values()])))
