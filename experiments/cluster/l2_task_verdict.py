#!/usr/bin/env python3
"""Verdict for a parameterised level_2 sweep (run_l2_task.sh).

Tests the single-obstacle hypothesis. Every level_1 task names ONE hazard and AEGIS beat us
on crash in all three (+3.8/+6.9/+4.2 against noise bands +-0.6/+-2.3/+-2.7). Every level_2
task names TWO, and on level_2 mango the gap vanished while the VLM named one obstacle in
all 150 cells. If that is causal, the same collapse should appear on the other level_2 tasks
-- including `apple`, whose two hazards are different KINDS, which separates "the VLM can
only pick one" from "the VLM confuses two identically-named objects".
"""
import json, pathlib, collections, statistics as st, sys

LR = pathlib.Path("${WORK_ROOT}/lora_dagger")
TAGS = sys.argv[1:] or ["apple", "onion"]
BDDL = {"apple": "white_yellow_mug_1 + wine_bottle_1 (DIFFERENT kinds)",
        "onion": "wine_bottle_1 + wine_bottle_2 (same kind)",
        "full":  "red_coffee_mug_1 + red_coffee_mug_2 (same kind, mango)"}

for tag in TAGS:
    root = LR / ("l2_" + tag)
    if not root.is_dir():
        print(f"[{tag}] no directory {root}"); continue
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    guarded = collections.Counter(); unusable = 0; hosts = set()
    for d in sorted(root.glob("*/off*_r*")):
        f = d / "result.json"
        if not f.is_file(): continue
        try: r = json.load(open(f))
        except Exception: continue
        if r.get("status") != "pass": unusable += 1; continue
        arm = d.parent.name; rep = d.name.split("_r")[1]
        md = r.get("metric_decomposition") or {}
        sr = int((r.get("successes") or 0) > 0); cc = float(md.get("policy_induced_cc") or 0)
        per[arm][rep].append((sr, cc))
        hf = d / "host.txt"
        if hf.is_file(): hosts.add(hf.read_text().strip())
        if arm == "aegis":
            srd = d / "srd.jsonl"
            if srd.exists():
                for line in srd.open(errors="ignore"):
                    try: x = json.loads(line)
                    except Exception: continue
                    if x.get("type") == "vlsa_audit" and x.get("obstacle_name"):
                        guarded[str(x["obstacle_name"])] += 1; break
    n_cells = sum(len(v) for a in per.values() for v in a.values())
    print(f"\n=== level_2 {tag}: {BDDL.get(tag,'?')} ===")
    print(f"    {n_cells} usable cells, {unusable} unusable, hosts={sorted(hosts)}")
    print("    arm         SR            crash         safe-fail   polCC")
    summ = {}
    for arm in ("base", "aegis", "ours"):
        reps = per.get(arm) or {}
        if not reps: print(f"    {arm:11s} no data"); continue
        srs, crs, sfs, ccs = [], [], [], []
        for rep, cells in sorted(reps.items()):
            n = len(cells)
            srs.append(sum(s for s, _ in cells) / n * 50)
            crs.append(sum(1 for s, c in cells if s == 0 and c > 0) / n * 50)
            sfs.append(sum(1 for s, c in cells if s == 0 and c == 0) / n * 50)
            ccs.append(sum(c for _, c in cells))
        f = lambda v: (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0)
        m_sr, s_sr = f(srs); m_cr, s_cr = f(crs); m_sf, _ = f(sfs)
        summ[arm] = (m_cr, s_cr, m_sr)
        print(f"    {arm:11s} {m_sr:5.1f}+-{s_sr:4.2f}   {m_cr:5.1f}+-{s_cr:4.2f}      {m_sf:5.1f}   {st.mean(ccs):7.0f}   (n={len(srs)})")
    if guarded:
        print(f"    VLM guarded: {dict(guarded.most_common(4))}   <- two hazards exist; how many were named?")
    if "ours" in summ and "aegis" in summ:
        (oc, osd, osr), (ac, asd, asr) = summ["ours"], summ["aegis"]
        pooled = (osd**2 + asd**2) ** 0.5
        band = 2 * pooled
        d = oc - ac
        call = ("ours WINS" if d < -band else "AEGIS wins" if d > band else "tie within noise")
        print(f"    -> crash {oc:.1f} vs {ac:.1f}  delta {d:+.1f} (pooled sd {pooled:.2f}, band +-{band:.1f})  {call}")
        print(f"       SR {osr:.1f} vs {asr:.1f}")
print("\nlevel_1 reference (1 hazard each): AEGIS won all three, crash gaps +3.8 / +6.9 / +4.2")
print("level_2 mango (2 same-kind hazards): tie, crash 3.0 vs 3.3, band +-1.3")
