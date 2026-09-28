#!/usr/bin/env python3
"""CURRICULUM vs STATIC with the repeats treated as what they are: clustered, not independent.

The 235 cells per arm are 47 offsets x 5 sampler-advance repeats. A McNemar test over all
235 pairs assumes 235 independent trials and inflates z accordingly. The offset is the unit
of randomisation here -- repeats of the same offset are the same scene.

So: aggregate to one number per offset (successes out of 5, total polCC over 5), then pair
CURRICULUM against STATIC offset by offset. Reported with a sign test and a bootstrap over
offsets, neither of which assumes normality on 47 points.
"""
import json, pathlib, collections, statistics as st, random, math

LR = pathlib.Path("${WORK_ROOT}/lora_dagger/cu2_eval")

def load(arm):
    per = collections.defaultdict(list)
    for c in sorted((LR/arm).glob("off*_r*")):
        f = c/"result.json"
        if not f.is_file(): continue
        try: r = json.load(open(f))
        except Exception: continue
        if r.get("status") != "pass": continue
        md = r.get("metric_decomposition") or {}
        off = int(c.name.split("_r")[0][3:])
        per[off].append((int((r.get("successes") or 0) > 0), float(md.get("policy_induced_cc") or 0)))
    return per

cur, sta = load("CURRICULUM"), load("STATIC")
shared = sorted(set(cur) & set(sta))
print("=== CURRICULUM vs STATIC, clustered by offset ===")
print("offsets shared: %d   (repeats per offset: cur %s, sta %s)"
      % (len(shared),
         sorted({len(cur[o]) for o in shared}), sorted({len(sta[o]) for o in shared})))

d_sr, d_cc, wins, losses, ties = [], [], 0, 0, 0
for o in shared:
    cs = sum(s for s, _ in cur[o]) / len(cur[o])
    ss = sum(s for s, _ in sta[o]) / len(sta[o])
    cc_c = sum(c for _, c in cur[o]) / len(cur[o])
    cc_s = sum(c for _, c in sta[o]) / len(sta[o])
    d_sr.append(cs - ss); d_cc.append(cc_c - cc_s)
    if cs > ss: wins += 1
    elif cs < ss: losses += 1
    else: ties += 1

print("\nper-offset success rate, CURRICULUM vs STATIC:")
print("  CURRICULUM better on %d offsets | STATIC better on %d | tied on %d"
      % (wins, losses, ties))
n = wins + losses
if n:
    # two-sided sign test, exact
    p = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n * 2
    p = min(1.0, p)
    print("  sign test on the %d non-tied offsets: p = %.3f  %s"
          % (n, p, "significant" if p < 0.05 else "NOT significant"))

def boot(vals, label, lower_is_better=False):
    random.seed(17)
    means = []
    for _ in range(20000):
        s = [vals[random.randrange(len(vals))] for _ in vals]
        means.append(sum(s) / len(s))
    means.sort()
    lo, hi = means[int(0.025 * len(means))], means[int(0.975 * len(means))]
    crosses = lo <= 0 <= hi
    print("  %-28s mean %+7.2f   95%% CI [%+.2f, %+.2f]  %s"
          % (label, st.mean(vals), lo, hi,
             "includes 0 -> NOT significant" if crosses else "excludes 0 -> significant"))
    return not crosses

print("\nbootstrap over the %d offsets (20000 resamples):" % len(shared))
sig_sr = boot(d_sr, "delta success rate")
sig_cc = boot(d_cc, "delta polCC per cell")

print("\n=== VERDICT ===")
if sig_sr or sig_cc:
    print("  Staged release shows a measurable effect on at least one metric.")
else:
    print("  No metric separates the arms once repeats are clustered by offset.")
    print()
    print("  SCOPE -- this is NOT evidence that curriculum learning does not work.")
    print("  The two arms' training files differ on 159 of 6535 rows (2.4%); the other")
    print("  97.6% are byte-identical. Composition of the training set:")
    print("      NO_RISK   5496 rows  84%   identical in both arms")
    print("      S1_early   709 rows  11%   identical in both arms")
    print("      S2_mid     330 rows   5%   159 of these differ  <- the whole variable")
    print("  and S3_emergency is dropped outright (DROP_STAGES) because its target actions")
    print("  are the crash-causing ones, so the steepest rung is missing by construction.")
    print()
    print("  Correct reading: THIS staging, with a 2.4% signal difference and only two")
    print("  rungs, produces no measurable effect. Testing 'easy to hard' properly needs a")
    print("  steeper gradient -- e.g. binning on |delta| or steps_to_first_risk directly")
    print("  (E11 measured |delta| rising monotonically 0.1818 / 0.2664 / 0.3764 / 0.5485).")
print("\n  (Naive per-cell McNemar over all 235 pairs gave z=2.40; it treats 5 repeats of")
print("   one scene as 5 independent trials, which they are not.)")
