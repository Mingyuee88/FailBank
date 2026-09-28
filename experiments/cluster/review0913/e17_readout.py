#!/usr/bin/env python3
"""E17: does the quiet-anchor result depend on the host or on the adapter lineage?
Five arms, L1-T3, all pinned to HOST_B. Per-offset mean over sampler conditions,
then a two-sided exact sign test over offsets."""
import json, math, os
from collections import defaultdict

R = "${WORK_ROOT}/lora_dagger/review0913/e17/l1_t3"
ARMS = ("base", "q00_orig", "q20_orig", "q00_s1", "q20_s1")


def load(arm):
    per = defaultdict(list)
    for off in range(50):
        for a in range(2):
            p = "%s/%s/off%d_r%d/result.json" % (R, arm, off, a)
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
    return w, l, t, min(1.0, sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n) * 2.0)


d = {a: load(a) for a in ARMS}
print("--- arm means (L1-T3, 50 offsets x 2 conditions, all on HOST_B) ---")
for a in ARMS:
    m = d[a]
    print("  %-9s n=%d  SR %6.1f   CC_policy %7.2f" % (
        a, len(m), 100 * sum(v[0] for v in m.values()) / len(m),
        sum(v[1] for v in m.values()) / len(m)))

print("\n--- paired contrasts (q20 - q00 within each lineage) ---")
for tag, q00, q20 in (("no-seed lineage (appendix)", "q00_orig", "q20_orig"),
                      ("seed-1 lineage (E15)", "q00_s1", "q20_s1")):
    offs = sorted(set(d[q00]) & set(d[q20]))
    sr = sign_test([(d[q20][o][0], d[q00][o][0]) for o in offs], True)
    cc = sign_test([(d[q20][o][1], d[q00][o][1]) for o in offs], False)
    dsr = 100 * (sum(d[q20][o][0] for o in offs) - sum(d[q00][o][0] for o in offs)) / len(offs)
    dcc = (sum(d[q20][o][1] for o in offs) - sum(d[q00][o][1] for o in offs)) / len(offs)
    print("  %-28s n=%d" % (tag, len(offs)))
    print("      SR  %+6.1f pts   %2d/%2d/%2d  p=%.4f" % (dsr, sr[0], sr[1], sr[2], sr[3]))
    print("      CC  %+6.2f       %2d/%2d/%2d  p=%.4f" % (dcc, cc[0], cc[1], cc[2], cc[3]))

print("\n--- lineage effect at fixed lambda_q ---")
for tag, a0, a1 in (("q00: seed1 - noseed", "q00_orig", "q00_s1"),
                    ("q20: seed1 - noseed", "q20_orig", "q20_s1")):
    offs = sorted(set(d[a0]) & set(d[a1]))
    sr = sign_test([(d[a1][o][0], d[a0][o][0]) for o in offs], True)
    dsr = 100 * (sum(d[a1][o][0] for o in offs) - sum(d[a0][o][0] for o in offs)) / len(offs)
    print("  %-22s SR %+6.1f pts  %2d/%2d/%2d p=%.4f" % (tag, dsr, sr[0], sr[1], sr[2], sr[3]))
