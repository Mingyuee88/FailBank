#!/usr/bin/env python3
"""E15 Stage B readout: lambda_q=0.2 (q20s1) vs lambda_q=0 (ncs1), seed 1, ten static tasks.

Protocol, as pre-registered: average each offset over its sampler conditions, then run a
two-sided exact sign test over offsets within a task. Ten-task aggregates weight tasks
equally. BRS uses the ten-task mean SR and policy-induced CC against the base arm.
"""
import json, math, os, sys
from collections import defaultdict

L = "${WORK_ROOT}/lora_dagger"
O = L + "/review0913"
TASKS = [
    ("L1-T0 Apple",  L + "/multi_t0",      2),
    ("L1-T1 Lemon",  L + "/multi_t1",      2),
    ("L1-T2 Mango",  L + "/multi_t2",      2),
    ("L1-T3 Onion",  L + "/multi_t3",      2),
    ("L1-T4 Tomato", L + "/multi_t4",      2),
    ("L2-T0 Apple",  L + "/l2_apple",      3),
    ("L2-T1 Lemon",  O + "/e10/l2_lemon",  3),
    ("L2-T2 Mango",  L + "/l2_full",       3),
    ("L2-T3 Onion",  L + "/l2_onion",      3),
    ("L2-T4 Tomato", O + "/e10/l2_tomato", 3),
]
ARMS = ("base", "ncs1", "q20s1")


def load(root, arm, reps):
    per = defaultdict(list)
    for off in range(50):
        for a in range(reps):
            p = "%s/%s/off%d_r%d/result.json" % (root, arm, off, a)
            if not os.path.exists(p):
                continue
            r = json.load(open(p))
            if r.get("status") != "pass":
                continue
            md = r.get("metric_decomposition") or {}
            per[off].append((float(r["successes"]), float(md["policy_induced_cc"])))
    return {o: (sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v))
            for o, v in per.items() if v}


def sign_test(pairs, higher_is_better):
    w = l = t = 0
    for a, b in pairs:
        if a == b: t += 1
        elif (a > b) == higher_is_better: w += 1
        else: l += 1
    n = w + l
    if n == 0: return w, l, t, 1.0
    k = min(w, l)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n) * 2.0
    return w, l, t, min(1.0, p)


rows, means = [], {a: {"sr": [], "cc": []} for a in ARMS}
print("%-14s %-28s %-28s" % ("", "SR (%)", "CC_policy"))
print("%-14s %7s %7s %7s  %-13s %7s %7s %7s  %-13s" % (
    "task", "base", "q00", "q20", "sign q20-q00", "base", "q00", "q20", "sign q20-q00"))
for name, root, reps in TASKS:
    d = {a: load(root, a, reps) for a in ARMS}
    offs = sorted(set(d["ncs1"]) & set(d["q20s1"]))
    m = {a: (100 * sum(d[a][o][0] for o in d[a]) / len(d[a]),
             sum(d[a][o][1] for o in d[a]) / len(d[a])) for a in ARMS}
    for a in ARMS:
        means[a]["sr"].append(m[a][0]); means[a]["cc"].append(m[a][1])
    sr = sign_test([(d["q20s1"][o][0], d["ncs1"][o][0]) for o in offs], True)
    cc = sign_test([(d["q20s1"][o][1], d["ncs1"][o][1]) for o in offs], False)
    rows.append((name, len(offs), m, sr, cc))
    print("%-14s %7.1f %7.1f %7.1f  %2d/%2d/%2d p=%.3f  %7.2f %7.2f %7.2f  %2d/%2d/%2d p=%.3f" % (
        name, m["base"][0], m["ncs1"][0], m["q20s1"][0], sr[0], sr[1], sr[2], sr[3],
        m["base"][1], m["ncs1"][1], m["q20s1"][1], cc[0], cc[1], cc[2], cc[3]))

print("\n--- ten-task means (task-equal weighting) ---")
agg = {a: (sum(means[a]["sr"]) / 10.0, sum(means[a]["cc"]) / 10.0) for a in ARMS}
for a in ARMS:
    print("  %-6s SR %6.2f   CC_policy %7.3f" % (a, agg[a][0], agg[a][1]))


def brs(sr, cc, sr_b, cc_b):
    sr, sr_b, cc, cc_b = sr / 100.0, sr_b / 100.0, cc, cc_b
    return math.exp(-0.5 * ((1 - sr) / (1 - sr_b) + (cc / cc_b if cc_b > 0 else 0.0)))


for a in ARMS:
    print("  %-6s BRS %.4f" % (a, brs(agg[a][0], agg[a][1], agg["base"][0], agg["base"][1])))

d_sr = agg["q20s1"][0] - agg["ncs1"][0]
d_cc = agg["q20s1"][1] - agg["ncs1"][1]
print("\n--- pre-registered decision rule ---")
print("  delta SR  (q20 - q00) = %+.2f points   (require >= -1.0)" % d_sr)
print("  delta CC  (q20 - q00) = %+.3f          (require <  0)" % d_cc)
go = (d_cc < 0) and (d_sr >= -1.0)
print("  DECISION: %s" % ("PROCEED to Stage C" if go else "STOP, report as is"))
