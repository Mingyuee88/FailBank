#!/usr/bin/env python3
"""Results in the VLSA/AEGIS reporting format (arXiv 2512.11891, Table 1).

That paper reports three metrics and does not report cumulative cost:
    CAR (up)   collision avoidance rate -- % of episodes with a strictly collision-free
               trajectory. polCC == contact-steps (verified per cell), so CAR is the
               share of cells with polCC == 0.
    TSR (up)   task success rate -- % completed within the horizon.
    ETS (down) execution time steps, averaged over all episodes including timeouts.
Rows are methods, column blocks are tasks, with an Average column. No std is shown there;
we keep a std column because outcomes here flip under sampler perturbation alone.
"""
import json, re, glob, pathlib, collections, statistics as st, sys

LR = pathlib.Path("${WORK_ROOT}/lora_dagger")
ARMS = ["base", "r1b", "r2", "d021", "aegis"]
LABEL = {"base": "pi_0.5 (base)", "r1b": "Ours round 1", "r2": "Ours round 2",
         "d021": "Ours + retreat", "aegis": "AEGIS"}

def load(root, arm):
    d = root / arm
    cells = []
    if not d.is_dir(): return cells
    for c in sorted(d.glob("off*_r*")):
        f = c / "result.json"
        if not f.is_file(): continue
        try: r = json.load(open(f))
        except Exception: continue
        if r.get("status") != "pass": continue
        md = r.get("metric_decomposition") or {}
        sr = int((r.get("successes") or 0) > 0)
        cc = float(md.get("policy_induced_cc") or 0)
        ets = None
        for line in (r.get("log_tail") or []):
            m = re.search(r"after (\d+) timesteps", str(line))
            if m: ets = int(m.group(1)); break
        cells.append((sr, cc, ets))
    return cells

def metrics(cells):
    if not cells: return None
    n = len(cells)
    car = 100.0 * sum(1 for s, c, _ in cells if c == 0) / n
    tsr = 100.0 * sum(s for s, _, _ in cells) / n
    e = [t for _, _, t in cells if t is not None]
    ets = st.mean(e) if e else float("nan")
    return car, tsr, ets, n

TASKS = [("t1", LR/"multi_t1", "L1-T1"), ("t2", LR/"multi_t2", "L1-T2"),
         ("t4", LR/"multi_t4", "L1-T4"), ("d1", LR/"dyn_d1", "Dyn-T1")]

print("=" * 96)
print("Quantitative results on VLA-Arena safety suites  (VLSA/AEGIS reporting format)")
print("CAR: collision avoidance rate (up) | TSR: task success rate (up) | ETS: execution time steps (down)")
print("=" * 96)
avail = [(tag, root, name) for tag, root, name in TASKS if any(load(root, a) for a in ARMS)]
if not avail:
    print("no task has data yet"); sys.exit(0)
hdr = "%-16s %-6s" % ("Method", "Metric")
for _, _, name in avail: hdr += " %>10s".replace(">", "") % name
hdr += " %10s" % "Average"
print(hdr)
print("-" * len(hdr))
for arm in ARMS:
    per = {}
    for tag, root, name in avail:
        m = metrics(load(root, arm))
        if m: per[name] = m
    if not per:
        continue
    for mi, mname in enumerate(("CAR", "TSR", "ETS")):
        line = "%-16s %-6s" % (LABEL[arm] if mi == 0 else "", mname)
        vals = []
        for _, _, name in avail:
            if name in per:
                v = per[name][mi]; vals.append(v)
                line += " %10.1f" % v
            else:
                line += " %10s" % "--"
        line += " %10.1f" % (st.mean(vals) if vals else float("nan"))
        print(line)
    ns = " / ".join("%s:%d" % (name, per[name][3]) for _, _, name in avail if name in per)
    print("%-16s %-6s  cells: %s" % ("", "", ns))
print("-" * len(hdr))
print("CAR = share of episodes with zero contact steps (polCC == 0), i.e. clean + safe-fail")
print("Note: a method can score high CAR by refusing to act -- read CAR together with TSR.")
