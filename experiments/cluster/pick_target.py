"""Compare AEGIS and our curriculum policy per Arena task, both under repeated measurement,
and pick the task where AEGIS is weakest."""
import json, glob, pathlib, statistics as st, collections
LR = pathlib.Path("${WORK_ROOT}/lora_dagger")
E8 = {"L1t1": dict(sr=44, cc=0, crash=0, safe=6),
      "L1t3": dict(sr=23, cc=407, crash=3, safe=24),
      "L1t4": dict(sr=47, cc=399, crash=2, safe=1)}
BASE_T3 = dict(sr=36, cc=1701, crash=13, safe=1)

def agg(root, task):
    runs = collections.defaultdict(dict)
    for d in sorted(glob.glob(str(root / task / "off*_r*"))):
        p = pathlib.Path(d)
        off = p.name.split("_r")[0].replace("off", ""); a = int(p.name.split("_r")[1])
        f = p / "result.json"
        if not f.exists(): continue
        r = json.load(open(f)); md = r.get("metric_decomposition") or {}
        runs[a][off] = (int((r.get("successes") or 0) > 0), float(md.get("policy_induced_cc") or 0))
    rows = []
    for a in sorted(runs):
        c = runs[a]
        if len(c) < 45: continue
        rows.append((sum(v[0] for v in c.values()),
                     sum(1 for v in c.values() if v[0] == 0 and v[1] > 0),
                     sum(1 for v in c.values() if v[0] == 0 and v[1] == 0),
                     sum(v[1] for v in c.values())))
    if not rows: return None
    return dict(n=len(rows),
                sr=st.mean([r[0] for r in rows]), sr_sd=st.pstdev([r[0] for r in rows]),
                crash=st.mean([r[1] for r in rows]), crash_sd=st.pstdev([r[1] for r in rows]),
                safe=st.mean([r[2] for r in rows]), cc=st.mean([r[3] for r in rows]))

print("%-6s %-24s %14s %14s %10s %9s" % ("task", "arm", "SR", "crash", "safe-fail", "polCC"))
weak = []
for task in ("L1t1", "L1t3", "L1t4"):
    a = agg(LR / "aegis_rep", task)
    o = agg(LR / "xfer", task)
    e = E8[task]
    if a:
        print("%-6s %-24s %6.1f+-%-6.2f %6.1f+-%-6.2f %10.1f %9.0f  (n=%d)"
              % (task, "AEGIS repeated", a["sr"], a["sr_sd"], a["crash"], a["crash_sd"],
                 a["safe"], a["cc"], a["n"]))
    print("%-6s %-24s %6d%-8s %6d%-8s %10d %9.0f" % (task, "AEGIS E8 single run",
          e["sr"], "", e["crash"], "", e["safe"], e["cc"]))
    if task == "L1t3":
        print("%-6s %-24s %6d%-8s %6d%-8s %10d %9.0f" % (task, "base (no shield)",
              BASE_T3["sr"], "", BASE_T3["crash"], "", BASE_T3["safe"], BASE_T3["cc"]))
    if o:
        print("%-6s %-24s %6.1f+-%-6.2f %6.1f+-%-6.2f %10.1f %9.0f  (n=%d)"
              % (task, "ours (curriculum pi_2)", o["sr"], o["sr_sd"], o["crash"], o["crash_sd"],
                 o["safe"], o["cc"], o["n"]))
    if a and o:
        gap = o["crash"] - a["crash"]
        pooled = ((a["crash_sd"]**2 + o["crash_sd"]**2)/2) ** 0.5
        verdict = "WE WIN" if gap < 0 else ("tie (within noise)" if abs(gap) < 2*pooled else "AEGIS wins")
        print("       -> crash gap %+.1f (pooled sd %.2f, noise band +-%.1f)  %s"
              % (gap, pooled, 2*pooled, verdict))
        weak.append((a["crash"] + a["safe"], task, a, o))
    print()
if weak:
    weak.sort(reverse=True)
    print("AEGIS is weakest (crash + safe-fail) on: %s" % weak[0][1])
