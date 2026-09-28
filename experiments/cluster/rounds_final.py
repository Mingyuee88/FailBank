#!/usr/bin/env python3
"""Rounds of self-evolution at a MATCHED training budget.

The first attempt at this comparison was confounded: r1 trained 100 steps, r2 trained 400,
because run_curriculum.sh and run_curriculum2.sh had different STEPS defaults. r1b retrains
round 1 on its own R1 records at 400 steps, so r1b vs r2 isolates the one variable that
matters -- which policy generated the training data.

    base   pi0, no failure-bank training
    r1     original round 1  (100 steps)   kept for reference only
    r1b    round 1 retrained (400 steps)   <- use this one
    r2     round 2           (400 steps)   trained on failures r1 produced

Round 3 is absent by measurement, not omission: its training was refused by the quiet-drift
guard (3364 rows against R2's 6535, same 400 steps -> overfit past the 0.05 limit).
"""
import json, pathlib, collections, statistics as st, random, math

root = pathlib.Path("${WORK_ROOT}/lora_dagger/rounds2")
LABEL = {"base": "pi0, zero rounds", "r1": "round 1 @100 steps (confounded)",
         "r1b": "round 1 @400 steps", "r2": "round 2 @400 steps"}

def load(arm):
    per = collections.defaultdict(list)
    d = root / arm
    if not d.is_dir(): return per
    for c in sorted(d.glob("off*_r*")):
        f = c / "result.json"
        if not f.is_file(): continue
        try: r = json.load(open(f))
        except Exception: continue
        if r.get("status") != "pass": continue
        md = r.get("metric_decomposition") or {}
        per[int(c.name.split("_r")[0][3:])].append(
            (int((r.get("successes") or 0) > 0), float(md.get("policy_induced_cc") or 0)))
    return per

arms = {a: load(a) for a in ("base", "r1", "r1b", "r2")}
print("=== self-evolution rounds, clustered by offset ===")
for a in ("base", "r1", "r1b", "r2"):
    per = arms[a]
    cells = [x for v in per.values() for x in v]
    if not cells:
        print("  %-4s (no data yet)" % a); continue
    byrep = collections.defaultdict(list)
    for off, v in per.items():
        for i, cell in enumerate(v): byrep[i].append(cell)
    srs, ccs, crs = [], [], []
    for rep, v in sorted(byrep.items()):
        n = len(v); srs.append(sum(s for s,_ in v)/n*50); ccs.append(sum(c for _,c in v))
        crs.append(sum(1 for s,c in v if s==0 and c>0)/n*50)
    sd = lambda v: st.pstdev(v) if len(v)>1 else 0.0
    clean = sum(1 for s,c in cells if s==1 and c==0)
    print("  %-4s offsets %2d cells %3d  SR %5.1f+-%4.2f  crash %5.1f+-%4.2f  polCC %6.0f  clean %3d   %s"
          % (a, len(per), len(cells), st.mean(srs), sd(srs), st.mean(crs), sd(crs),
             st.mean(ccs), clean, LABEL[a]))

def compare(a, b, note=""):
    pa, pb = arms[a], arms[b]
    shared = sorted(set(pa) & set(pb))
    if len(shared) < 10:
        print("\n%s -> %s: only %d shared offsets, skipping" % (a, b, len(shared))); return
    d_sr, d_cc, w, l = [], [], 0, 0
    for o in shared:
        sa = sum(s for s,_ in pa[o])/len(pa[o]); sb = sum(s for s,_ in pb[o])/len(pb[o])
        ca = sum(c for _,c in pa[o])/len(pa[o]); cb = sum(c for _,c in pb[o])/len(pb[o])
        d_sr.append(sb-sa); d_cc.append(cb-ca)
        if sb > sa: w += 1
        elif sb < sa: l += 1
    n = w + l
    p = min(1.0, sum(math.comb(n,k) for k in range(min(w,l)+1))/2**n*2) if n else 1.0
    def boot(v):
        random.seed(17); m=[]
        for _ in range(20000):
            s=[v[random.randrange(len(v))] for _ in v]; m.append(sum(s)/len(s))
        m.sort(); return m[int(.025*len(m))], m[int(.975*len(m))]
    lo_sr, hi_sr = boot(d_sr); lo_cc, hi_cc = boot(d_cc)
    sig_sr = not (lo_sr <= 0 <= hi_sr); sig_cc = not (lo_cc <= 0 <= hi_cc)
    print("\n%s -> %s  (%d offsets)%s" % (a, b, len(shared), "   " + note if note else ""))
    print("  per-offset: %s better on %d, %s better on %d, tied %d   sign p=%.3f"
          % (b, w, a, l, len(shared)-n, p))
    print("  dSR    %+6.3f  CI [%+.3f, %+.3f]  %s" % (st.mean(d_sr), lo_sr, hi_sr,
          "SIGNIFICANT" if sig_sr else "not significant"))
    print("  dpolCC %+6.1f  CI [%+.1f, %+.1f]  %s" % (st.mean(d_cc), lo_cc, hi_cc,
          "SIGNIFICANT" if sig_cc else "not significant"))
    return sig_sr, sig_cc

compare("base", "r1b", "does one round of failure-bank training help?")
res = compare("r1b", "r2", "<-- THE MULTI-ROUND CLAIM, budget now matched")
compare("base", "r2")
print()
print("For reference only (confounded by training budget, 100 vs 400 steps):")
compare("r1", "r2", "IGNORE -- unequal training")
print()
if res and not any(res):
    print("VERDICT: with the budget matched, round 2 does NOT improve on round 1.")
    print("The earlier r1->r2 gain was the extra training, not the self-evolution.")
elif res:
    print("VERDICT: round 2 improves on round 1 at equal training budget --")
    print("the gain is attributable to training on failures round 1 itself produced.")
