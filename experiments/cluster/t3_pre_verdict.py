#!/usr/bin/env python3
"""Verdict for the pre-contact warning-line sweep (run_t3_pre.sh).

Answers two things:
  1. does firing BEFORE contact reduce the grazing cost that makes up ~99% of our polCC
     gap to AEGIS on level_2 (1118 of 1129 points), and at what cost in SR;
  2. what distance pi_2 actually reaches, so the warning line stops being a guess.
"""
import re, glob, json, collections, statistics as st

PAT = re.compile(
    r"T3P_RESULT arm=(\S+) off=(\d+) adv=(\d+) host=(\S+) sr=(\d+) polcc=(\S+) "
    r"fires=(\d+) releases=(\d+) retreat_steps=(\d+) dmin=(\S+) contact_steps=(\S+) below=(\{.*\})")
ARMS = ("off", "d019", "d021", "d023")
LINE = {"off": "layer off (same-host baseline)", "d019": "warn < 0.19",
        "d021": "warn < 0.21", "d023": "warn < 0.23"}

rows = collections.defaultdict(list); hosts = set(); unusable = 0
for fn in glob.glob("${WORK_ROOT}/lora_dagger/t3_pre/batch/T3pre.o*"):
    for line in open(fn, errors="ignore"):
        if "T3P_UNUSABLE" in line: unusable += 1; continue
        m = PAT.search(line)
        if m:
            rows[m.group(1)].append(m.groups()); hosts.add(m.group(4))

print("=== PRE-CONTACT WARNING LINE (L1t3, 12 hot offsets x 2 repeats) ===")
print("hosts: %s%s   unusable cells: %d"
      % (sorted(hosts), "" if len(hosts) <= 1 else "   <-- NOT PINNED, deltas unreliable", unusable))
print()
print("arm    n   SR     crash   polCC    fires  releases  retreat_steps")
summ = {}
for arm in ARMS:
    v = rows.get(arm) or []
    if not v:
        print("%-6s no data" % arm); continue
    n = len(v)
    sr = sum(int(x[4]) for x in v) / n
    cc = sum(float(x[5]) for x in v) / n
    crash = sum(1 for x in v if int(x[4]) == 0 and float(x[5]) > 0) / n
    summ[arm] = (sr, crash, cc)
    print("%-6s %2d  %5.2f  %5.2f  %7.1f  %5d  %6d  %8d  %s"
          % (arm, n, sr, crash, cc, sum(int(x[6]) for x in v),
             sum(int(x[7]) for x in v), sum(int(x[8]) for x in v), LINE[arm]))

if "off" in summ:
    b_sr, b_cr, b_cc = summ["off"]
    print("\ndeltas vs the same-host `off` baseline:")
    for arm in ARMS[1:]:
        if arm not in summ: continue
        sr, cr, cc = summ[arm]
        print("  %-5s  SR %+5.2f   crash %+5.2f   polCC %+7.1f  (%.0f%% of baseline cost)"
              % (arm, sr - b_sr, cr - b_cr, cc - b_cc, 100 * cc / b_cc if b_cc else 0))

print("\n=== distance calibration from the `off` arm ===")
v = rows.get("off") or []
dm = sorted(float(x[9]) for x in v if x[9] not in ("None", ""))
if dm:
    print("  per-cell minimum cost_pair_min_distance:")
    print("    p10 %.4f   median %.4f   p90 %.4f   max %.4f"
          % (dm[max(0, len(dm)//10)], st.median(dm), dm[min(len(dm)-1, 9*len(dm)//10)], dm[-1]))
agg = collections.Counter(); contact_tot = 0; steps_seen = 0
for x in v:
    try:
        agg.update({k: int(c) for k, c in json.loads(x[11].replace("'", '"')).items()})
    except Exception:
        pass
    if x[10] not in ("None", ""):
        contact_tot += int(x[10])
if agg:
    print("  steps below each candidate line (summed over off-arm cells):")
    for k in sorted(agg, key=float):
        print("     < %-6s %6d steps" % (k, agg[k]))
    print("     contact steps total: %d" % contact_tot)
    print("  -> a line with far more steps than contact steps fires constantly and will cost SR;")
    print("     one with fewer cannot pre-empt the graze at all.")
