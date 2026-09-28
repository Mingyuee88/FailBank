# Pre-declared metrics (fixed before computing):
#  SSR  = episode succeeds AND policy_induced_cc == 0
#  CPS  = total policy_induced_cc / number of successes (lower is better)
#  Pareto: A dominates B if SR_A >= SR_B and pCC_A <= pCC_B with at least one strict
import sys
sys.argv = ["x"]
exec(open("analyze_review.py").read().split("# ------------------------------------------------------------------ E1")[0])

def ssr_cells(A):
    out = {}
    for o, cells in A.items():
        out[o] = [dict(s=float(c["s"] > 0 and (c["pcc"] or 0) == 0), pcc=c["pcc"], occ=c["occ"]) for c in cells]
    return out

def stats(A):
    cells = [c for v in A.values() for c in v]
    n = len(cells); succ = sum(c["s"] for c in cells)
    ssr = 100 * sum(1 for c in cells if c["s"] > 0 and (c["pcc"] or 0) == 0) / n
    tot = sum((c["pcc"] or 0) for c in cells)
    cps = tot / succ if succ else float("inf")
    return dict(sr=100 * succ / n, pcc=tot / n, ssr=ssr, cps=cps)

def avg_seeds(arms):
    return {o: [c for A in arms for c in A[o]] for o in range(50) if all(o in A for A in arms)}

def pareto(a, b):
    ge = a["sr"] >= b["sr"] and a["pcc"] <= b["pcc"]
    strict = a["sr"] > b["sr"] or a["pcc"] < b["pcc"]
    le = a["sr"] <= b["sr"] and a["pcc"] >= b["pcc"]
    strict2 = a["sr"] < b["sr"] or a["pcc"] > b["pcc"]
    return "FB dominates" if ge and strict else ("FB dominated" if le and strict2 else "trade-off")

print("| 表 | 任务 | 臂 | SR | policy CC | SSR | 每次成功代价 |")
print("|---|---|---|---|---|---|---|")
tests = []
rows = []
for t in range(5):
    b = arm(f"{LR}/multi_t{t}/base/off{{o}}_r{{a}}/result.json", 50, 2)
    ae = arm(f"{LR}/multi_t{t}/aegis/off{{o}}_r{{a}}/result.json", 50, 2)
    fb = arm(f"{LR}/multi_t{t}/nocurr/off{{o}}_r{{a}}/result.json", 50, 2)
    rows.append((f"L1 T{t}", b, ae, fb))
for tag, sub in (("L2 apple", "l2_apple"), ("L2 mango", "l2_full"), ("L2 onion", "l2_onion")):
    b = arm(f"{LR}/{sub}/base/off{{o}}_r{{a}}/result.json", 50, 3)
    ae = arm(f"{LR}/{sub}/aegis/off{{o}}_r{{a}}/result.json", 50, 3)
    fb = avg_seeds([arm(f"{LR}/{sub}/ncs{s}/off{{o}}_r{{a}}/result.json", 50, 3) for s in (1, 2, 3)])
    rows.append((tag, b, ae, fb))
for tag, b, ae, fb in rows:
    sb, sa, sf = stats(b), stats(ae), stats(fb)
    for name, s in (("base", sb), ("AEGIS", sa), ("FailBank", sf)):
        print(f"| π0.5 | {tag} | {name} | {s['sr']:.1f} | {s['pcc']:.2f} | {s['ssr']:.1f} | {s['cps']:.2f} |")
    w1 = paired(ssr_cells(fb), ssr_cells(ae), "s"); w2 = paired(ssr_cells(fb), ssr_cells(b), "s")
    tests.append((tag, w1, w2, pareto(sf, sa), pareto(sf, sb), sf, sa, sb))
print()
print("| 任务 | SSR FailBank vs AEGIS | SSR FailBank vs base | Pareto vs AEGIS | Pareto vs base |")
print("|---|---|---|---|---|")
for tag, w1, w2, pa, pb, *_ in tests:
    print(f"| {tag} | {w1[0]}/{w1[1]}/{w1[2]} p={w1[3]:.4f} | {w2[0]}/{w2[1]}/{w2[2]} p={w2[3]:.4f} | {pa} | {pb} |")

# pooled SSR over task-offset units (L1 five tasks; L2 three tasks)
for label, idx in (("L1 五任务", range(0, 5)), ("L2 三任务", range(5, 8)), ("全部 8 任务", range(0, 8))):
    P = {"b": {}, "a": {}, "f": {}}
    for i in idx:
        tag, b, ae, fb = rows[i]
        for key, A in (("b", b), ("a", ae), ("f", fb)):
            for o, v in ssr_cells(A).items():
                P[key][(i, o)] = v
    fa = paired(P["f"], P["a"], "s"); fbb = paired(P["f"], P["b"], "s")
    m = lambda D: 100 * mean([c["s"] for v in D.values() for c in v])
    print(f"POOLED {label}: SSR base {m(P['b']):.1f} AEGIS {m(P['a']):.1f} FailBank {m(P['f']):.1f} | FB vs AEGIS {fa[0]}/{fa[1]} p={fa[3]:.4f} | FB vs base {fbb[0]}/{fbb[1]} p={fbb[3]:.4f}")

print("\nπ0 table (no AEGIS arm):")
for t, name in enumerate(["apple", "lemon", "mango", "onion", "tomato"]):
    b = arm(f"{LR}/pi0_eval2/base_t{t}_off{{o}}/result.json", 50, 1)
    q = arm(f"{LR}/pi0_eval2/ours_t{t}_off{{o}}/result.json", 50, 1)
    sb, sq = stats(b), stats(q); w = paired(ssr_cells(q), ssr_cells(b), "s")
    print(f"| π0 | {name} | base SR {sb['sr']:.1f} SSR {sb['ssr']:.1f} | FailBank SR {sq['sr']:.1f} SSR {sq['ssr']:.1f} | SSR {w[0]}/{w[1]}/{w[2]} p={w[3]:.4f} |")
