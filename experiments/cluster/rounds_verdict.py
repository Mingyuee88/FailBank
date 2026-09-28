#!/usr/bin/env python3
"""Zero / one / two rounds, clustered by offset (repeats of one scene aren't independent)."""
import json, pathlib, collections, statistics as st, random, math

root = pathlib.Path("${WORK_ROOT}/lora_dagger/rounds")
def load(arm):
    per = collections.defaultdict(list)
    for c in sorted((root/arm).glob("off*_r*")):
        f = c/"result.json"
        if not f.is_file(): continue
        try: r = json.load(open(f))
        except Exception: continue
        if r.get("status") != "pass": continue
        md = r.get("metric_decomposition") or {}
        per[int(c.name.split("_r")[0][3:])].append(
            (int((r.get("successes") or 0) > 0), float(md.get("policy_induced_cc") or 0)))
    return per

arms = {a: load(a) for a in ("base", "p1", "p2")}
print("=== rounds of self-evolution, clustered by offset ===")
for a, per in arms.items():
    cells = [x for v in per.values() for x in v]
    srs, ccs, crs = [], [], []
    byrep = collections.defaultdict(list)
    for off, v in per.items():
        for i, cell in enumerate(v): byrep[i].append(cell)
    for rep, v in sorted(byrep.items()):
        n = len(v); srs.append(sum(s for s,_ in v)/n*50); ccs.append(sum(c for _,c in v))
        crs.append(sum(1 for s,c in v if s==0 and c>0)/n*50)
    sd = lambda v: st.pstdev(v) if len(v)>1 else 0.0
    clean = sum(1 for s,c in cells if s==1 and c==0)
    print("  %-5s offsets %2d  SR %5.1f+-%4.2f  crash %5.1f+-%4.2f  polCC %6.0f  clean %3d"
          % (a, len(per), st.mean(srs), sd(srs), st.mean(crs), sd(crs), st.mean(ccs), clean))

def compare(a, b):
    pa, pb = arms[a], arms[b]
    shared = sorted(set(pa) & set(pb))
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
    print("\n%s -> %s  (%d offsets)" % (a, b, len(shared)))
    print("  per-offset: %s better on %d, %s better on %d, tied %d   sign p=%.3f"
          % (b, w, a, l, len(shared)-n, p))
    print("  dSR    %+6.3f  CI [%+.3f, %+.3f]  %s" % (st.mean(d_sr), lo_sr, hi_sr,
          "significant" if not (lo_sr <= 0 <= hi_sr) else "not significant"))
    print("  dpolCC %+6.1f  CI [%+.1f, %+.1f]  %s" % (st.mean(d_cc), lo_cc, hi_cc,
          "significant" if not (lo_cc <= 0 <= hi_cc) else "not significant"))

compare("base", "p1")
compare("p1", "p2")
compare("base", "p2")
