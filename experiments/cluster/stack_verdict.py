#!/usr/bin/env python3
"""Verdict for pi_2 + AEGIS in-loop (l2_stack) against the three l2_full arms.

The question is whether the two are complementary. From the level_2 mango decomposition:
    bucket              ours          aegis
    grazing success  105 @ 11.9    24 @  5.5
    crash              9 @ 172.4   10 @ 154.1   <- already at parity
    clean success         36          108       <- the entire polCC gap (1118/1129)
    safe-fail              0            8
If the stack works, it should keep pi_2's SR and zero safe-fail while moving cells from the
"grazing success" column into "clean success".
"""
import json, pathlib, collections, statistics as st

LR = pathlib.Path("${WORK_ROOT}/lora_dagger")
SRC = {"base": LR/"l2_full/base", "aegis": LR/"l2_full/aegis", "ours": LR/"l2_full/ours",
       "ours+aegis": LR/"l2_stack/ours_aegis"}

def load(d):
    out = collections.defaultdict(list); hosts = set(); bad = 0
    if not d.is_dir(): return out, hosts, bad
    for c in sorted(d.glob("off*_r*")):
        f = c/"result.json"
        if not f.is_file(): continue
        try: r = json.load(open(f))
        except Exception: continue
        if r.get("status") != "pass": bad += 1; continue
        md = r.get("metric_decomposition") or {}
        rep = c.name.split("_r")[1]
        out[rep].append((int((r.get("successes") or 0) > 0), float(md.get("policy_induced_cc") or 0)))
        hf = c/"host.txt"
        if hf.is_file(): hosts.add(hf.read_text().strip())
    return out, hosts, bad

print("=== level_2 mango: does stacking the shield on pi_2 buy clean successes? ===")
print("arm          n    SR          crash       safe-fail  polCC   | clean  graze(mean)  crash(mean)")
summ = {}
for arm, d in SRC.items():
    per, hosts, bad = load(d)
    cells = [c for v in per.values() for c in v]
    if not cells:
        print("%-12s (no data yet)" % arm); continue
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
    summ[arm] = (st.mean(srs), st.mean(crs), st.mean(ccs), clean, len(graze))
    print("%-12s %3d  %5.1f+-%4.2f  %4.1f+-%4.2f    %4.1f   %6.0f   | %4d   %3d @%5.1f  %3d @%6.1f%s"
          % (arm, len(cells), st.mean(srs), sd(srs), st.mean(crs), sd(crs), st.mean(sfs),
             st.mean(ccs), clean, len(graze), st.mean(graze) if graze else 0,
             len(crash), st.mean(crash) if crash else 0,
             "" if len(hosts) <= 1 else "  HOSTS=%s" % sorted(hosts)))
    if bad: print("             (%d unusable cells excluded)" % bad)

if "ours+aegis" in summ and "ours" in summ and "aegis" in summ:
    s = summ["ours+aegis"]; o = summ["ours"]; a = summ["aegis"]
    print("\nstack vs its two ingredients:")
    print("  vs ours   SR %+5.1f  crash %+4.1f  polCC %+6.0f  clean %+4d" % (s[0]-o[0], s[1]-o[1], s[2]-o[2], s[3]-o[3]))
    print("  vs aegis  SR %+5.1f  crash %+4.1f  polCC %+6.0f  clean %+4d" % (s[0]-a[0], s[1]-a[1], s[2]-a[2], s[3]-a[3]))
    print("\n  complementary would mean: SR at ours' level, polCC toward aegis', clean up sharply.")
