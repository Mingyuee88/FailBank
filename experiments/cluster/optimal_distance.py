#!/usr/bin/env python3
"""Distance to the ideal point, with the base policy on the plot and weight sensitivity shown.

Two honest requirements shape this:
  * the base policy belongs on the chart -- what matters for the paper is how far each
    method moved from where it started, not only where it ended;
  * the ranking depends on how SR and CC are weighted, and any single weighting is a
    choice. Reporting one number without the sensitivity would be picking the weighting
    that flatters us.

Normalisation: SR is a percentage so (100-SR)/100 is its own scale. CC has no natural
ceiling, so it is scaled by the BASE policy's CC -- i.e. "what fraction of the untrained
model's cost remains". That makes 1.0 mean "no improvement over base" and 0 mean "no cost",
which is the comparison the paper actually needs to make.
"""
import json, pathlib, math
import numpy as np

LR = pathlib.Path("${WORK_ROOT}/lora_dagger")
ARMS = [("base (pi0.5-ft)", "multi_%s", "base"),
        ("ours (pi_2)",     "multi_%s", "r2"),
        ("AEGIS",           "multi_%s", "aegis")]
TASKS = ("t1", "t2", "t4")

def agg(root, arm):
    cells = []
    d = root / arm
    if not d.is_dir(): return None
    for c in sorted(d.glob("off*_r*")):
        f = c / "result.json"
        if not f.is_file(): continue
        try: r = json.load(open(f))
        except Exception: continue
        if r.get("status") != "pass": continue
        cells.append((int((r.get("successes") or 0) > 0),
                      float((r.get("metric_decomposition") or {}).get("policy_induced_cc") or 0)))
    if not cells: return None
    return (100*np.mean([c[0] for c in cells]), np.mean([c[1] for c in cells]))

pts = {}
for lbl, pat, arm in ARMS:
    vals = [agg(LR/(pat % t), arm) for t in TASKS]
    vals = [v for v in vals if v]
    if len(vals) == len(TASKS):
        pts[lbl] = (np.mean([v[0] for v in vals]), np.mean([v[1] for v in vals]))

print("平均 over L1-T1/T2/T4 (100 cells each)")
print("  %-18s %7s %7s" % ("", "SR", "CC"))
for k, (sr, cc) in pts.items():
    print("  %-18s %7.1f %7.1f" % (k, sr, cc))
base_cc = pts["base (pi0.5-ft)"][1]
print()
print("归一化：dSR = (100-SR)/100 ;  dCC = CC / CC_base = 剩下多少基座的代价")
print("  %-18s %8s %8s" % ("", "dSR", "dCC"))
for k, (sr, cc) in pts.items():
    print("  %-18s %8.3f %8.3f" % (k, (100-sr)/100, cc/base_cc))
print()
print("到理想点 (SR 100, CC 0) 的加权距离，w = CC 的权重")
print("  %-18s %8s %8s %8s %8s %8s" % ("", "w=0.0", "w=0.25", "w=0.5", "w=0.75", "w=1.0"))
for k, (sr, cc) in pts.items():
    row = "  %-18s" % k
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        d = math.hypot((1-w)*((100-sr)/100), w*(cc/base_cc))
        row += " %8.3f" % d
    print(row)
print()
print("  w=0   只看 SR      w=1   只看 CC")
best = {}
for w in (0.0, 0.25, 0.5, 0.75, 1.0):
    b = min(pts, key=lambda k: math.hypot((1-w)*((100-pts[k][0])/100), w*(pts[k][1]/base_cc)))
    best[w] = b
print("  各权重下最近的方法：%s" % {("w=%.2f" % w): b for w, b in best.items()})
print()
print("相对基座的改进（这是 base 必须在图上的原因）")
bs, bc = pts["base (pi0.5-ft)"]
for k, (sr, cc) in pts.items():
    if k == "base (pi0.5-ft)": continue
    print("  %-18s SR %+5.1f 分   CC %+5.1f (%+.0f%%)" % (k, sr-bs, cc-bc, 100*(cc-bc)/bc))
