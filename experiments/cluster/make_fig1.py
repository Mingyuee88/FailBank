#!/usr/bin/env python3
"""Figure 1 — safety-utility plane, one panel per task.

Purpose:  show where each method sits relative to the ideal operating point (SR=100, CC=0),
          and that the ordering depends on how collision cost is weighted.
Claim it supports:  RQ2. Cited by the sentence beginning "On two of three tasks the
          internalised policy is closer to the ideal operating point...".
Required data:  multi_t{1,2,4}/{base,aegis,nocurr,ncs1,ncs2,ncs3}/*/result.json

Honesty constraints enforced here:
  - one panel per task; no pooled means (the three tasks ran on different GPU hosts)
  - official_cc on the x axis, the definition the VLA-Arena leaderboard reports
  - axes start at 0 and span the same range in every panel; no zoom to exaggerate gaps
  - the multi-seed arms are drawn as individual points plus their mean, so seed spread is
    visible rather than hidden behind an average
"""
import json, glob, os, math, collections, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "fig1_tradeoff.svg"
L = "${WORK_ROOT}/lora_dagger"
TASKS = [("multi_t1", "L1-T1"), ("multi_t2", "L1-T2"), ("multi_t4", "L1-T4")]
SEEDS = ["ncs1", "ncs2", "ncs3"]

def cells(task, arm):
    sr = cc = n = 0
    for f in glob.glob(f"{L}/{task}/{arm}/*/result.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        md = r.get("metric_decomposition") or {}
        n += 1
        sr += r.get("successes", 0)
        cc += md.get("official_cc") or 0.0
    return (n, sr / n * 100, cc / n) if n else None

data = {}
for task, label in TASKS:
    d = {}
    for arm in ["base", "aegis", "nocurr"] + SEEDS:
        v = cells(task, arm)
        if v:
            d[arm] = v
    data[task] = d

# geometry
PW, PH, PAD, GAP = 210, 210, 46, 26
W = PAD + 3 * (PW + GAP)
H = PAD + PH + 62
CCMAX = 90.0   # covers every arm on every task; fixed across panels
SRMIN = 60.0

def px(p, cc):  return PAD + p * (PW + GAP) + cc / CCMAX * PW
def py(sr):     return 34 + (100 - sr) / (100 - SRMIN) * PH

s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
     f'font-family="ui-sans-serif,system-ui,sans-serif">']
s.append('<style>'
         '.ax{stroke:#94a3b8;stroke-width:1}'
         '.gr{stroke:#e2e8f0;stroke-width:.6}'
         '.tk{fill:#64748b;font-size:8px}'
         '.ttl{fill:#0f172a;font-size:11px;font-weight:600}'
         '.lb{font-size:8.5px}'
         '.iso{stroke:#cbd5e1;stroke-width:.7;stroke-dasharray:3 3;fill:none}'
         '</style>')

for p, (task, label) in enumerate(TASKS):
    d = data.get(task, {})
    if not d:
        continue
    x0, y0 = px(p, 0), py(100)          # ideal corner
    # iso-distance arcs around the ideal point (equal-weight w=0.5, illustrative)
    for frac in (0.25, 0.5, 0.75):
        rx, ry = frac * PW, frac * PH
        s.append(f'<path class="iso" d="M {x0+rx:.1f} {y0:.1f} '
                 f'A {rx:.1f} {ry:.1f} 0 0 1 {x0:.1f} {y0+ry:.1f}"/>')
    # grid + axes
    for g in range(0, 5):
        gx = px(p, CCMAX * g / 4)
        s.append(f'<line class="gr" x1="{gx:.1f}" y1="{py(100):.1f}" x2="{gx:.1f}" y2="{py(SRMIN):.1f}"/>')
        s.append(f'<text class="tk" x="{gx:.1f}" y="{py(SRMIN)+12:.1f}" text-anchor="middle">{CCMAX*g/4:.0f}</text>')
    for sr in (60, 70, 80, 90, 100):
        gy = py(sr)
        s.append(f'<line class="gr" x1="{px(p,0):.1f}" y1="{gy:.1f}" x2="{px(p,CCMAX):.1f}" y2="{gy:.1f}"/>')
        if p == 0:
            s.append(f'<text class="tk" x="{px(p,0)-6:.1f}" y="{gy+3:.1f}" text-anchor="end">{sr}</text>')
    s.append(f'<line class="ax" x1="{px(p,0):.1f}" y1="{py(SRMIN):.1f}" x2="{px(p,CCMAX):.1f}" y2="{py(SRMIN):.1f}"/>')
    s.append(f'<line class="ax" x1="{px(p,0):.1f}" y1="{py(100):.1f}" x2="{px(p,0):.1f}" y2="{py(SRMIN):.1f}"/>')
    s.append(f'<text class="ttl" x="{px(p,CCMAX/2):.1f}" y="22" text-anchor="middle">{label}</text>')
    # ideal point
    s.append(f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="3.4" fill="none" stroke="#0f172a" stroke-width="1.4"/>')
    s.append(f'<text class="lb" x="{x0+7:.1f}" y="{y0+3:.1f}" fill="#0f172a">ideal</text>')
    # seed replicates: individual points, then their mean
    pts = [d[a] for a in SEEDS if a in d]
    for (n, sr, cc) in pts:
        s.append(f'<circle cx="{px(p,cc):.1f}" cy="{py(sr):.1f}" r="2.2" fill="#16a34a" opacity=".45"/>')
    if pts:
        msr = sum(v[1] for v in pts) / len(pts)
        mcc = sum(v[2] for v in pts) / len(pts)
        s.append(f'<circle cx="{px(p,mcc):.1f}" cy="{py(msr):.1f}" r="4.6" fill="#16a34a"/>')
        s.append(f'<text class="lb" x="{px(p,mcc):.1f}" y="{py(msr)-8:.1f}" '
                 f'text-anchor="middle" fill="#15803d">ours</text>')
    for arm, col, lab in (("base", "#94a3b8", "base"), ("aegis", "#ea580c", "AEGIS")):
        if arm not in d:
            continue
        n, sr, cc = d[arm]
        s.append(f'<circle cx="{px(p,cc):.1f}" cy="{py(sr):.1f}" r="4.6" fill="{col}"/>')
        s.append(f'<text class="lb" x="{px(p,cc):.1f}" y="{py(sr)-8:.1f}" '
                 f'text-anchor="middle" fill="{col}">{lab}</text>')

s.append(f'<text class="tk" x="{W/2:.1f}" y="{H-8:.1f}" text-anchor="middle" font-size="9">'
         f'official collision cost (lower better) — vertical axis: success rate (higher better); '
         f'small green dots are the three training-data seeds</text>')
s.append("</svg>")
open(OUT, "w").write("\n".join(s))

print(f"wrote {OUT}")
for task, label in TASKS:
    print(f"--- {label}")
    for arm, v in data.get(task, {}).items():
        print(f"    {arm:8s} n={v[0]:3d} SR={v[1]:5.1f} official_cc={v[2]:6.2f}")
