#!/usr/bin/env python3
"""Verdict on clipping, judged against the correct baseline and on the right statistic.

Baseline is the r2 arm: same checkpoint, no safety layer. (An earlier comparison used
multi_*/base by mistake, which unsets the checkpoint and is therefore the pi0.5-finetuned
base model -- that made the layer look far better than it is.)

The layer's damage was a long tail: CC change median ~0 on every task, mean +40 to +52,
with five cells carrying most of it. So the numbers that decide whether clipping worked are
the tail, not the mean: how many fired cells got worse, and how big the worst ones are.

Pass condition, stated up front: on the cells where it fires, CC must not increase and SR
must not drop. Anything else means the layer still costs more than it returns.
"""
import json, pathlib
import numpy as np

LR = pathlib.Path("${WORK_ROOT}/lora_dagger")
ARMS = [("no layer (r2)", "multi_%s", "r2"),
        ("cloned CBF", "cbfcl_%s", "dhead"),
        ("clip 0.20", "clipc20_%s", "dhead"),
        ("clip 0.10", "clipc10_%s", "dhead"),
        ("AEGIS", "multi_%s", "aegis")]

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

print("%-16s %7s %7s %7s %6s" % ("", "SR", "CC", "CFR", "n"))
means = {}
for tag, name in (("t1","L1-T1"), ("t2","L1-T2"), ("t4","L1-T4")):
    print(name)
    base = load(LR/("multi_"+tag), "r2")
    for lbl, pat, arm in ARMS:
        a = load(LR/(pat % tag), arm)
        if not a:
            print("  %-14s   (无数据)" % lbl); continue
        sr = 100*np.mean([v["sr"] for v in a.values()])
        cc = np.mean([v["cc"] for v in a.values()])
        cfr = 100*np.mean([v["cc"] == 0 for v in a.values()])
        means.setdefault(lbl, []).append((sr, cc, cfr))
        line = "  %-14s %7.1f %7.1f %7.1f %6d" % (lbl, sr, cc, cfr, len(a))
        shared = sorted(set(base) & set(a))
        fired = [k for k in shared if a[k]["fired"] > 0]
        if fired and lbl not in ("no layer (r2)", "AEGIS"):
            d = np.array([a[k]["cc"] - base[k]["cc"] for k in fired])
            ds = np.array([a[k]["sr"] - base[k]["sr"] for k in fired])
            line += "   | fired %3d  dCC mean %+6.1f med %+5.1f  worst %+5.0f  dSR %+.0f%%" % (
                len(fired), d.mean(), np.median(d), d.max(), 100*ds.mean())
        print(line)
    print()
print("平均（三任务）")
for lbl, v in means.items():
    if len(v) == 3:
        print("  %-14s SR %5.1f  CC %5.1f  CFR %5.1f" % (
            lbl, np.mean([x[0] for x in v]), np.mean([x[1] for x in v]), np.mean([x[2] for x in v])))
print()
print("判据：在触发的格上 dCC <= 0 且 dSR >= 0。任何一项不满足，该形态不成立。")
