#!/usr/bin/env python3
"""Full-set verdict for the pre-contact warning line, AEGIS on the same machine.

Reports the bucket decomposition that actually matters: the polCC gap to AEGIS is made of
grazing successes, not collisions (level_2: 105 grazing cells @11.9 vs AEGIS's 24 @5.5
accounts for 1118 of the 1129-point gap). A warning line is supposed to move cells from
`graze` into `clean` without moving `clean` into `safe-fail`.
"""
import re, glob, collections, statistics as st

PAT = re.compile(r"T3F_RESULT arm=(\S+) off=(\d+) adv=(\d+) host=(\S+) sr=(\d+) polcc=(\S+) "
                 r"fires=(\d+) releases=(\d+) dmin=(\S+) contact_steps=(\S+)")
ARMS = ("off", "d021", "d023", "aegis")
DESC = {"off": "pi_2, layer off", "d021": "pi_2 + warn<0.21", "d023": "pi_2 + warn<0.23",
        "aegis": "published shield (same host)"}

rows = collections.defaultdict(lambda: collections.defaultdict(list))
hosts = set(); unusable = collections.Counter()
for fn in glob.glob("${WORK_ROOT}/lora_dagger/t3_full/batch/T3full.o*"):
    for line in open(fn, errors="ignore"):
        if "T3F_UNUSABLE" in line:
            m = re.search(r"arm=(\S+).*status=(\S+)", line)
            if m: unusable[(m.group(1), m.group(2))] += 1
            continue
        m = PAT.search(line)
        if m:
            g = m.groups(); rows[g[0]][g[2]].append((int(g[4]), float(g[5]), int(g[6])))
            hosts.add(g[3])

print("=== L1t3 FULL SET (50 offsets x 2 repeats), pre-contact warning line ===")
print("hosts: %s%s" % (sorted(hosts), "" if len(hosts) <= 1 else "  <-- NOT PINNED"))
if unusable:
    print("unusable: %s" % dict(unusable))
print()
print("arm     n    SR          crash       safe-fail   polCC   | clean  graze(mean)  crash(mean)  fires")
summ = {}
for arm in ARMS:
    per = rows.get(arm) or {}
    cells = [c for v in per.values() for c in v]
    if not cells:
        print("%-7s (no data)" % arm); continue
    srs, crs, sfs, ccs = [], [], [], []
    for rep, v in sorted(per.items()):
        n = len(v)
        srs.append(sum(s for s, _, _ in v)/n*50); ccs.append(sum(c for _, c, _ in v))
        crs.append(sum(1 for s, c, _ in v if s == 0 and c > 0)/n*50)
        sfs.append(sum(1 for s, c, _ in v if s == 0 and c == 0)/n*50)
    sd = lambda v: st.pstdev(v) if len(v) > 1 else 0.0
    clean = sum(1 for s, c, _ in cells if s == 1 and c == 0)
    graze = [c for s, c, _ in cells if s == 1 and c > 0]
    crash = [c for s, c, _ in cells if s == 0 and c > 0]
    summ[arm] = dict(sr=st.mean(srs), crash=st.mean(crs), sf=st.mean(sfs), cc=st.mean(ccs),
                     sd_cr=sd(crs), clean=clean, graze=len(graze))
    print("%-7s %3d  %5.1f+-%4.2f  %4.1f+-%4.2f    %4.1f   %6.0f   | %4d  %3d @%5.1f  %3d @%6.1f  %5d"
          % (arm, len(cells), st.mean(srs), sd(srs), st.mean(crs), sd(crs), st.mean(sfs),
             st.mean(ccs), clean, len(graze), st.mean(graze) if graze else 0,
             len(crash), st.mean(crash) if crash else 0, sum(f for _, _, f in cells)))
    print("        %s" % DESC[arm])

if "off" in summ:
    b = summ["off"]
    print("\nvs pi_2 baseline (same host):")
    for arm in ("d021", "d023", "aegis"):
        if arm not in summ: continue
        a = summ[arm]
        print("  %-6s SR %+5.1f   crash %+5.1f   polCC %+6.0f (%3.0f%%)   clean %+4d   graze %+4d"
              % (arm, a["sr"]-b["sr"], a["crash"]-b["crash"], a["cc"]-b["cc"],
                 100*a["cc"]/b["cc"] if b["cc"] else 0, a["clean"]-b["clean"], a["graze"]-b["graze"]))
if "aegis" in summ:
    a = summ["aegis"]
    print("\nvs AEGIS (same host) -- the comparison that decides the paper:")
    for arm in ("off", "d021", "d023"):
        if arm not in summ: continue
        o = summ[arm]
        pooled = (o["sd_cr"]**2 + a["sd_cr"]**2) ** 0.5; band = 2*pooled
        d = o["crash"] - a["crash"]
        call = "OURS WINS" if d < -band else "AEGIS wins" if d > band else "tie within noise"
        print("  %-6s crash %4.1f vs %4.1f  delta %+5.1f (band +-%.1f)  %-16s SR %5.1f vs %5.1f  safe-fail %4.1f vs %4.1f"
              % (arm, o["crash"], a["crash"], d, band, call, o["sr"], a["sr"], o["sf"], a["sf"]))
