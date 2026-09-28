#!/usr/bin/env python3
"""CURRICULUM vs STATIC -- does staging the release order contribute anything?

This is the only comparison that isolates the claim "easy to hard". Both arms (from
run_curriculum2.sh, STEPS=400 BATCH=32 -- real training strength, not smoke):

    CURRICULUM   phase1 = s1   (225 deltas, S1_early)       -> fold -> phase2 = s1s2
    STATIC       phase1 = s1s2 (290 deltas, S1_early+S2_mid) -> fold -> phase2 = s1s2

Identical phase-2 data, identical starting checkpoint, identical number of optimizer
resets, identical record count (3657). The ONLY difference is whether phase 1 withheld
S2_mid. If the two arms tie, staged release has no independent contribution and the
method's core claim does not hold.

Paired by cell: the same offset+repeat under both arms, which removes cell difficulty as a
variance source. Outcomes are known to be fragile (24/24 flipped under sampler perturbation
alone), so the paired count and the per-repeat spread both matter more than the means.
"""
import json, pathlib, collections, statistics as st

LR = pathlib.Path("${WORK_ROOT}/lora_dagger/cu2_eval")

def load(arm):
    out = {}
    hosts = set()
    for c in sorted((LR/arm).glob("off*_r*")):
        f = c/"result.json"
        if not f.is_file(): continue
        try: r = json.load(open(f))
        except Exception: continue
        if r.get("status") != "pass": continue
        md = r.get("metric_decomposition") or {}
        out[c.name] = (int((r.get("successes") or 0) > 0), float(md.get("policy_induced_cc") or 0))
        hf = c/"host.txt"
        if hf.is_file(): hosts.add(hf.read_text().strip())
    return out, hosts

cur, h1 = load("CURRICULUM")
sta, h2 = load("STATIC")
print("=== CURRICULUM vs STATIC (the release-order isolation) ===")
print("hosts: %s" % sorted(h1 | h2))
print("cells: CURRICULUM %d, STATIC %d" % (len(cur), len(sta)))

def summarise(name, d):
    per = collections.defaultdict(list)
    for k, v in d.items():
        per[k.split("_r")[1]].append(v)
    srs, crs, sfs, ccs = [], [], [], []
    for rep, v in sorted(per.items()):
        n = len(v)
        srs.append(sum(s for s, _ in v)/n*50); ccs.append(sum(c for _, c in v)/n*50)
        crs.append(sum(1 for s, c in v if s == 0 and c > 0)/n*50)
        sfs.append(sum(1 for s, c in v if s == 0 and c == 0)/n*50)
    sd = lambda v: st.pstdev(v) if len(v) > 1 else 0.0
    cells = list(d.values())
    clean = sum(1 for s, c in cells if s == 1 and c == 0)
    graze = sum(1 for s, c in cells if s == 1 and c > 0)
    print("  %-11s SR %5.1f+-%4.2f   crash %4.1f+-%4.2f   safe-fail %4.1f   polCC %6.0f  | clean %3d graze %3d  (%d repeats)"
          % (name, st.mean(srs), sd(srs), st.mean(crs), sd(crs), st.mean(sfs), st.mean(ccs),
             clean, graze, len(per)))
    return dict(sr=st.mean(srs), sd_sr=sd(srs), cr=st.mean(crs), sd_cr=sd(crs), cc=st.mean(ccs))

print()
c = summarise("CURRICULUM", cur)
s = summarise("STATIC", sta)

shared = sorted(set(cur) & set(sta))
print("\npaired on %d identical cells (same offset, same repeat):" % len(shared))
both_ok = sum(1 for k in shared if cur[k][0] == 1 and sta[k][0] == 1)
cur_only = sum(1 for k in shared if cur[k][0] == 1 and sta[k][0] == 0)
sta_only = sum(1 for k in shared if cur[k][0] == 0 and sta[k][0] == 1)
neither  = sum(1 for k in shared if cur[k][0] == 0 and sta[k][0] == 0)
print("  both succeed %3d | CURRICULUM only %3d | STATIC only %3d | neither %3d"
      % (both_ok, cur_only, sta_only, neither))
# McNemar on the discordant pairs
n_d = cur_only + sta_only
if n_d:
    import math
    z = abs(cur_only - sta_only) / math.sqrt(n_d)
    print("  discordant %d, split %d/%d -> z = %.2f %s"
          % (n_d, cur_only, sta_only, z,
             "(significant at 0.05)" if z > 1.96 else "(NOT significant)"))
dc = [cur[k][1] - sta[k][1] for k in shared]
print("  paired polCC difference (CURRICULUM - STATIC): mean %+.1f  median %+.1f"
      % (st.mean(dc), st.median(dc)))

print("\n=== VERDICT ===")
band_sr = 2 * ((c["sd_sr"]**2 + s["sd_sr"]**2) ** 0.5)
band_cr = 2 * ((c["sd_cr"]**2 + s["sd_cr"]**2) ** 0.5)
d_sr, d_cr = c["sr"] - s["sr"], c["cr"] - s["cr"]
print("  SR    %+5.1f  (noise band +-%.1f)  -> %s" % (d_sr, band_sr,
      "CURRICULUM better" if d_sr > band_sr else "STATIC better" if d_sr < -band_sr else "TIE"))
print("  crash %+5.1f  (noise band +-%.1f)  -> %s" % (d_cr, band_cr,
      "CURRICULUM better" if d_cr < -band_cr else "STATIC better" if d_cr > band_cr else "TIE"))
print("\n  A tie on both means staged release has no independent contribution,")
print("  and the 'easy to hard' claim cannot be supported by this experiment.")
