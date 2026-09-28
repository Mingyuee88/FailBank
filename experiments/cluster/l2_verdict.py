"""level_2: does AEGIS's one-obstacle design lose to a policy that never saw this level?"""
import json, glob, pathlib, statistics as st, collections
R = pathlib.Path("${WORK_ROOT}/lora_dagger/l2_full")
def agg(arm):
    runs = collections.defaultdict(dict)
    guard = collections.Counter(); near = collections.Counter()
    for d in sorted(glob.glob(str(R / arm / "off*_r*"))):
        p = pathlib.Path(d)
        off = p.name.split("_r")[0].replace("off", ""); a = int(p.name.split("_r")[1])
        f = p / "result.json"
        if not f.exists(): continue
        r = json.load(open(f)); md = r.get("metric_decomposition") or {}
        runs[a][off] = (int((r.get("successes") or 0) > 0), float(md.get("policy_induced_cc") or 0))
        srd = p / "srd.jsonl"
        if srd.exists():
            for line in srd.open():
                try: x = json.loads(line)
                except Exception: continue
                if x.get("type") == "vlsa_audit" and x.get("obstacle_name"):
                    guard[str(x["obstacle_name"])] += 1
                pre = x.get("pre_step_features") or {}
                if pre.get("cost_pair_min_name"): near[pre["cost_pair_min_name"]] += 1
    rows = []
    for a in sorted(runs):
        c = runs[a]
        if len(c) < 30: continue
        rows.append((sum(v[0] for v in c.values()),
                     sum(1 for v in c.values() if v[0] == 0 and v[1] > 0),
                     sum(1 for v in c.values() if v[0] == 0 and v[1] == 0),
                     sum(v[1] for v in c.values()), len(c)))
    if not rows: return None, guard, near
    return dict(n=len(rows), cells=rows[0][4],
                sr=st.mean([r[0] for r in rows]), sr_sd=st.pstdev([r[0] for r in rows]),
                crash=st.mean([r[1] for r in rows]), crash_sd=st.pstdev([r[1] for r in rows]),
                safe=st.mean([r[2] for r in rows]), cc=st.mean([r[3] for r in rows])), guard, near
print("safety_static_obstacles LEVEL 2, task 2 (mango), two hazards in the cost predicate")
print("pi_2 was trained on level_1 task 2 only -- level 2 is unseen.\n")
print("%-8s %14s %14s %10s %10s" % ("arm", "SR", "crash", "safe-fail", "polCC"))
res = {}
for arm in ("base", "aegis", "ours"):
    a, guard, near = agg(arm)
    res[arm] = a
    if not a:
        print("%-8s (incomplete)" % arm); continue
    print("%-8s %6.1f+-%-6.2f %6.1f+-%-6.2f %10.1f %10.0f   (n=%d repeats, %d cells)"
          % (arm, a["sr"], a["sr_sd"], a["crash"], a["crash_sd"], a["safe"], a["cc"], a["n"], a["cells"]))
    if arm == "aegis" and guard:
        print("         VLM named as the guarded obstacle: %s" % dict(guard.most_common(3)))
    if near:
        print("         nearest cost-pair seen: %s" % dict(near.most_common(3)))
if res.get("aegis") and res.get("ours"):
    o, g = res["ours"], res["aegis"]
    pooled = ((o["crash_sd"]**2 + g["crash_sd"]**2)/2) ** 0.5
    d = o["crash"] - g["crash"]
    print("\n--- ours vs AEGIS on level 2 ---")
    print("  crash %.1f vs %.1f   delta %+.1f  (pooled sd %.2f, noise band +-%.1f)"
          % (o["crash"], g["crash"], d, pooled, 2*pooled))
    print("  SR    %.1f vs %.1f   |  safe-fail %.1f vs %.1f  |  polCC %.0f vs %.0f"
          % (o["sr"], g["sr"], o["safe"], g["safe"], o["cc"], g["cc"]))
    verdict = "WE WIN on safety" if d < -2*pooled else ("tie within noise" if abs(d) <= 2*pooled else "AEGIS wins")
    print("  ->", verdict)
