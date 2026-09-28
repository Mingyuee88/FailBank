#!/usr/bin/env python3
"""Verdict for the contact-budget run. All five arms share one host.

Target: polCC <= aegis's, with SR at pi_2's own level (not bought with safe-fails).
The escape claim is checked directly via `escape_still_touching`: if the release+lift does
not break contact, the budget saves nothing and the arm must not be credited.
"""
import re, glob, collections, statistics as st

PAT = re.compile(r"BG_RESULT arm=(\S+) off=(\d+) adv=(\d+) host=(\S+) sr=(\d+) polcc=(\S+) "
                 r"budget_spent_at=(\S+) escape_steps=(\d+) escape_still_touching=(\d+) proj=(\d+)")
ARMS = ("off", "aegis", "sh", "b25", "sh_b25")
DESC = {"off": "pi_2 alone", "aegis": "published shield", "sh": "pi_2 + our shield",
        "b25": "pi_2 + contact budget 25", "sh_b25": "pi_2 + shield + budget 25"}

rows = collections.defaultdict(lambda: collections.defaultdict(list))
esc = collections.defaultdict(lambda: [0, 0, 0]); hosts = set(); bad = collections.Counter()
for fn in glob.glob("${WORK_ROOT}/lora_dagger/budget/batch/*"):
    for line in open(fn, errors="ignore"):
        if "BG_UNUSABLE" in line:
            m = re.search(r"arm=(\S+).*status=(\S+)", line)
            if m: bad[(m.group(1), m.group(2))] += 1
            continue
        m = PAT.search(line)
        if not m: continue
        g = m.groups()
        rows[g[0]][g[2]].append((int(g[4]), float(g[5])))
        hosts.add(g[3])
        e = esc[g[0]]
        if g[6] != "None": e[0] += 1
        e[1] += int(g[7]); e[2] += int(g[8])

print("=== CONTACT BUDGET on L1t3 (50 offsets x 2 repeats, one host) ===")
print("hosts: %s%s" % (sorted(hosts), "" if len(hosts) <= 1 else "  <-- NOT PINNED"))
if bad: print("unusable: %s" % dict(bad))
print()
print("arm       n    SR          crash       safe-fail  polCC | clean graze crash | budget")
summ = {}
for arm in ARMS:
    per = rows.get(arm) or {}
    cells = [c for v in per.values() for c in v]
    if not cells:
        print("%-9s (no data)" % arm); continue
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
    spent, esteps, still = esc[arm]
    summ[arm] = dict(sr=st.mean(srs), cr=st.mean(crs), cc=st.mean(ccs), sd=sd(crs), n=len(cells))
    b = ("spent %d/%d cells, %d escape steps, %d still touching (%.0f%%)"
         % (spent, len(cells), esteps, still, 100*still/esteps if esteps else 0)) if esteps else "-"
    print("%-9s %3d  %5.1f+-%4.2f  %4.1f+-%4.2f    %4.1f  %6.0f | %4d %5d %5d | %s"
          % (arm, len(cells), st.mean(srs), sd(srs), st.mean(crs), sd(crs), st.mean(sfs),
             st.mean(ccs), clean, len(graze), len(crash), b))
    print("          %s" % DESC[arm])

if "aegis" in summ and "off" in summ:
    a, o = summ["aegis"], summ["off"]
    print("\n=== TARGET: polCC <= %.0f (aegis) with SR >= %.1f (pi_2 alone) ===" % (a["cc"], o["sr"]))
    MIN_N = 60   # of 100; below this a few lucky cells can fake any verdict
    for arm in ("sh", "b25", "sh_b25"):
        if arm not in summ: continue
        s = summ[arm]
        if s["n"] < MIN_N:
            print("  %-8s polCC %6.0f   SR %5.1f   crash %4.1f   TOO FEW CELLS (%d/100) -- no verdict"
                  % (arm, s["cc"], s["sr"], s["cr"], s["n"]))
            continue
        if s["cc"] <= a["cc"] and s["sr"] >= o["sr"] - 1.0:
            v = "GOAL MET -- polCC beaten at full capability"
        elif s["cc"] <= a["cc"] and s["sr"] >= a["sr"]:
            v = "beats aegis on both, %.1f SR below pi_2" % (o["sr"] - s["sr"])
        elif s["cc"] <= a["cc"]:
            v = "polCC won but SR below aegis -- not a win"
        else:
            v = "polCC above aegis by %.0f" % (s["cc"] - a["cc"])
        print("  %-8s polCC %6.0f   SR %5.1f   crash %4.1f   %s" % (arm, s["cc"], s["sr"], s["cr"], v))
