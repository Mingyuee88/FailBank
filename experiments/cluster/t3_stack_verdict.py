#!/usr/bin/env python3
"""L1t3 verdict: does the stack also win on AEGIS's STRONGEST task?

level_2 mango showed ours+aegis at crash 0.7+-0.47 against AEGIS's 3.3+-0.47 (delta -2.6,
band +-1.3). But level_2 is where AEGIS is weakest -- two hazards, one VLM name. L1t3 has a
single hazard and is where AEGIS beat us by the widest level_1 margin (crash 1.4 vs 8.3).
If the stack wins here too, the result is about the combination, not about a blind spot.

Baselines `off` (pi_2 alone) and `aegis` come from the T3full run on the same machine.
"""
import re, glob, collections, statistics as st

def collect(pattern, pat, arm_idx=0, rep_idx=2, sr_idx=4, cc_idx=5, host_idx=3):
    rows = collections.defaultdict(lambda: collections.defaultdict(list)); hosts = set()
    for fn in glob.glob(pattern):
        for line in open(fn, errors="ignore"):
            m = pat.search(line)
            if m:
                g = m.groups()
                rows[g[arm_idx]][g[rep_idx]].append((int(g[sr_idx]), float(g[cc_idx])))
                hosts.add(g[host_idx])
    return rows, hosts

LR = "${WORK_ROOT}/lora_dagger"
P_FULL = re.compile(r"T3F_RESULT arm=(\S+) off=(\d+) adv=(\d+) host=(\S+) sr=(\d+) polcc=(\S+)")
P_STACK = re.compile(r"T3K_RESULT arm=(\S+) off=(\d+) adv=(\d+) host=(\S+) sr=(\d+) polcc=(\S+)")
a, h1 = collect(LR + "/t3_full/batch/T3full.o*", P_FULL)
b, h2 = collect(LR + "/t3_stack/batch/T3stack.o*", P_STACK)
rows = {**a, **b}; hosts = h1 | h2

print("=== L1t3 (AEGIS's strongest task): does the stack still win? ===")
print("hosts: %s%s\n" % (sorted(hosts), "" if len(hosts) <= 1 else "  <-- NOT PINNED"))
print("arm          n    SR          crash       safe-fail  polCC  | clean  graze(mean)  crash(mean)")
summ = {}
for arm in ("off", "d021", "d023", "aegis", "stack", "stack_d021"):
    per = rows.get(arm) or {}
    cells = [c for v in per.values() for c in v]
    if not cells:
        print("%-12s (no data)" % arm); continue
    srs, crs, sfs, ccs = [], [], [], []
    for rep, v in sorted(per.items()):
        n = len(v)
        srs.append(sum(s for s, _ in v)/n*50); ccs.append(sum(c for _, c in v))
        crs.append(sum(1 for s, c in v if s == 0 and c > 0)/n*50)
        sfs.append(sum(1 for s, c in v if s == 0 and c == 0)/n*50)
    sd = lambda v: st.pstdev(v) if len(v) > 1 else 0.0
    clean = sum(1 for s, c in cells if s == 1 and c == 0)
    graze = [c for s, c in cells if s == 1 and c > 0]
    crash = [c for s, c in cells if s == 0 and c > 0]
    summ[arm] = dict(sr=st.mean(srs), cr=st.mean(crs), sf=st.mean(sfs), cc=st.mean(ccs), sd=sd(crs))
    print("%-12s %3d  %5.1f+-%4.2f  %4.1f+-%4.2f    %4.1f   %6.0f  | %4d  %3d @%5.1f  %3d @%6.1f"
          % (arm, len(cells), st.mean(srs), sd(srs), st.mean(crs), sd(crs), st.mean(sfs),
             st.mean(ccs), clean, len(graze), st.mean(graze) if graze else 0,
             len(crash), st.mean(crash) if crash else 0))

if "aegis" in summ:
    a_ = summ["aegis"]
    print("\nvs AEGIS on the same machine:")
    for arm in ("off", "d021", "d023", "stack", "stack_d021"):
        if arm not in summ: continue
        o = summ[arm]
        pooled = (o["sd"]**2 + a_["sd"]**2) ** 0.5; band = 2*pooled
        d = o["cr"] - a_["cr"]
        call = "OURS WINS" if d < -band else "AEGIS wins" if d > band else "tie"
        print("  %-11s crash %4.1f vs %4.1f   delta %+5.1f (band +-%.1f)  %-11s  SR %5.1f vs %5.1f  polCC %5.0f vs %5.0f"
              % (arm, o["cr"], a_["cr"], d, band, call, o["sr"], a_["sr"], o["cc"], a_["cc"]))
print("\nlevel_2 mango reference: ours+aegis crash 0.7+-0.47 vs aegis 3.3+-0.47 (delta -2.6, band +-1.3)")
