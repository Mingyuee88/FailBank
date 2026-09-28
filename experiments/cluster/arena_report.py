#!/usr/bin/env python3
"""Results table in the VLSA/AEGIS layout, using VLA-Arena's own metrics.

Layout follows arXiv 2512.11891 Table 1 ("Quantitative results on the SafeLIBERO
benchmark"): rows are methods, each method spans several metric rows, column blocks are
tasks, plus an Average column.

Metrics are VLA-Arena's, not SafeLIBERO's -- the leaderboard reports SR and CC (cumulative
cost) for safety tasks, whereas VLSA's CAR/TSR/ETS belong to SafeLIBERO. Reporting the
host benchmark's own metrics keeps the numbers comparable to the leaderboard:
    SR  (up)   success rate, % of episodes completed within the horizon
    CC  (down) policy-induced cumulative cost. NOT official_cost: on several tasks the
               whole official cost is `initial_attributed_cost`, present at reset before
               the policy acts (e.g. teapots that topple on reset).
    CFR (up)   collision-free rate, % of episodes with CC == 0. This is VLSA's CAR
               computed on Arena's cost, given polCC == contact-steps.
    ETS (down) execution time steps, averaged over all episodes including timeouts.

A task where almost no cell has CC > 0 cannot separate methods; such tasks are flagged.
"""
import json, re, pathlib, collections, statistics as st, sys

LR = pathlib.Path("${WORK_ROOT}/lora_dagger")
ARMS = ["base", "r1b", "r2", "d021", "aegis"]
LABEL = {"base": "pi_0.5 (base)", "r1b": "Ours (round 1)", "r2": "Ours (round 2)",
         "d021": "Ours + retreat", "aegis": "AEGIS"}
TASKS = [("L1-T1", LR/"multi_t1"), ("L1-T2", LR/"multi_t2"), ("L1-T4", LR/"multi_t4"),
         ("Dyn-T0", LR/"dyn_d0"), ("Dyn-T1", LR/"dyn_d1")]

def load(root, arm):
    out = []
    d = root/arm
    if not d.is_dir(): return out
    for c in sorted(d.glob("off*_r*")):
        f = c/"result.json"
        if not f.is_file(): continue
        try: r = json.load(open(f))
        except Exception: continue
        if r.get("status") != "pass": continue
        md = r.get("metric_decomposition") or {}
        ep = (r.get("episodes") or [{}])[0]
        ets = None
        for line in (r.get("log_tail") or []):
            m = re.search(r"after (\d+) timesteps", str(line))
            if m: ets = int(m.group(1)); break
        out.append((int((r.get("successes") or 0) > 0),
                    float(md.get("policy_induced_cc") or 0),
                    float(ep.get("initial_attributed_cost") or 0), ets))
    return out

def agg(cells):
    if not cells: return None
    n = len(cells)
    sr = 100.0*sum(s for s,_,_,_ in cells)/n
    cc = st.mean([c for _,c,_,_ in cells])
    cfr = 100.0*sum(1 for _,c,_,_ in cells if c == 0)/n
    e = [t for _,_,_,t in cells if t is not None]
    return dict(SR=sr, CC=cc, CFR=cfr, ETS=st.mean(e) if e else float("nan"),
                n=n, costly=sum(1 for _,c,_,_ in cells if c > 0),
                init=sum(1 for _,_,i,_ in cells if i > 0))

avail = [(name, root) for name, root in TASKS if any(load(root, a) for a in ARMS)]
if not avail:
    print("no data yet"); sys.exit(0)

print("=" * (26 + 11*(len(avail)+1)))
print("Quantitative results on VLA-Arena safety suites")
print("SR/CFR: higher is better   CC/ETS: lower is better   CC = policy-induced cost only")
print("=" * (26 + 11*(len(avail)+1)))
hdr = "%-17s %-5s" % ("Method", "")
for name, _ in avail: hdr += "%11s" % name
hdr += "%11s" % "Average"
print(hdr); print("-"*len(hdr))

for arm in ARMS:
    per = {name: agg(load(root, arm)) for name, root in avail}
    if not any(per.values()): continue
    for mi, mname in enumerate(("SR", "CFR", "CC", "ETS")):
        line = "%-17s %-5s" % (LABEL[arm] if mi == 0 else "", mname)
        vals = []
        for name, _ in avail:
            a = per.get(name)
            if a: vals.append(a[mname]); line += "%11.1f" % a[mname]
            else: line += "%11s" % "--"
        line += "%11.1f" % (st.mean(vals) if vals else float("nan"))
        print(line)
    cells = "  ".join("%s n=%d" % (name, per[name]["n"]) for name,_ in avail if per.get(name))
    print("%-23s %s" % ("", cells))
print("-"*len(hdr))
print()
print("task discriminability check -- a task where few cells carry policy-induced cost")
print("cannot separate methods, whatever the numbers look like:")
for name, root in avail:
    a = agg(load(root, "base"))
    if not a: continue
    flag = "  <-- NOT DISCRIMINATIVE" if a["costly"] < 0.2*a["n"] else ""
    print("  %-8s base cells=%3d  cells with policy cost=%3d (%2.0f%%)  cells with reset cost=%3d%s"
          % (name, a["n"], a["costly"], 100.0*a["costly"]/a["n"], a["init"], flag))
