import sys, random, math
sys.argv = ["x"]
exec(open("analyze_review.py").read().split("# ------------------------------------------------------------------ E1")[0])
random.seed(0)

def avg_seeds(arms, noff=50):
    return {o: [c for A in arms for c in A[o]] for o in range(noff) if all(o in A for A in arms)}

def host_counts(root):
    c = {}
    for d in glob.glob(root + "/*"):
        h = d + "/host.txt"
        if os.path.exists(h):
            k = open(h).read().strip(); c[k] = c.get(k, 0) + 1
    return c

L2 = {}
print("## E10 读数\n")
for tid, tag, host in ((1, "lemon", "HOST_A"), (4, "tomato", "HOST_B")):
    root = f"{O}/e10/l2_{tag}"
    A = {a: arm(f"{root}/{a}/off{{o}}_r{{a}}/result.json", 50, 3) for a in ("base", "aegis", "ncs1", "ncs2", "ncs3")}
    fb = avg_seeds([A["ncs1"], A["ncs2"], A["ncs3"]])
    L2[tag] = (A["base"], A["aegis"], fb)
    print(f"### L2 T{tid} {tag}（{host}）")
    print("hosts:", {a: host_counts(f"{root}/{a}") for a in A})
    print("| 臂 | offsets | SR | official CC | policy CC |\n|---|---|---|---|---|")
    for a in ("base", "aegis", "ncs1", "ncs2", "ncs3"):
        s = summ(A[a]); print(f"| {a} | {s['n']} | {s['sr']:.1f} | {s['occ']:.2f} | {s['pcc']:.2f} |")
    s = summ(fb); srs = [summ(A[f'ncs{k}'])['sr'] for k in (1, 2, 3)]
    print(f"| ncs 三 seed 平均 | {s['n']} | {s['sr']:.1f} (SD {statistics.stdev(srs):.1f}) | {s['occ']:.2f} | {s['pcc']:.2f} |")
    print()
    print(HDR)
    for k in (1, 2, 3):
        print(row(f"ncs{k}", A[f"ncs{k}"], A["base"], "base"))
    print(row("ncs 三 seed 平均", fb, A["base"], "base"))
    print(row("ncs 三 seed 平均", fb, A["aegis"], "aegis"))
    print(row("aegis", A["aegis"], A["base"], "base"))
    floor = all(summ(A[a])["sr"] <= 5.0 for a in ("base", "ncs1", "ncs2", "ncs3"))
    print(f"\nFLOOR_RULE(base 与 ncs1-3 SR 均 ≤5%): {floor}\n")

# existing L2 tasks
for tag, sub in (("apple", "l2_apple"), ("mango", "l2_full"), ("onion", "l2_onion")):
    b = arm(f"{LR}/{sub}/base/off{{o}}_r{{a}}/result.json", 50, 3)
    ae = arm(f"{LR}/{sub}/aegis/off{{o}}_r{{a}}/result.json", 50, 3)
    fb = avg_seeds([arm(f"{LR}/{sub}/ncs{s}/off{{o}}_r{{a}}/result.json", 50, 3) for s in (1, 2, 3)])
    L2[tag] = (b, ae, fb)

print("### L2 汇总（ncs 三 seed 平均 vs base / aegis，任务×offset 为单位）\n")
print(HDR)
for label, tags in (("L2 五任务 T0–T4", ("apple", "lemon", "mango", "onion", "tomato")),
                    ("L2 四任务（去掉 lemon）", ("apple", "mango", "onion", "tomato")),
                    ("L2 原三任务 T0/T2/T3", ("apple", "mango", "onion")),
                    ("L2 新两任务 T1/T4", ("lemon", "tomato"))):
    P = {"b": {}, "a": {}, "f": {}}
    for tg in tags:
        b, ae, fb = L2[tg]
        for key, D in (("b", b), ("a", ae), ("f", fb)):
            for o, v in D.items():
                P[key][(tg, o)] = v
    print(row(label + " FailBank", P["f"], P["b"], "base"))
    print(row(label + " FailBank", P["f"], P["a"], "aegis"))

# Score B with bootstrap over offsets
def B(cells_arm, cells_base):
    fa = 1 - mean([c["s"] for c in cells_arm]); fbse = 1 - mean([c["s"] for c in cells_base])
    pa = mean([c["pcc"] for c in cells_arm]); pb = mean([c["pcc"] for c in cells_base])
    if fbse <= 0 or pb <= 0:
        return None
    return math.exp(-0.5 * (fa / fbse + pa / pb))

tasks = []
for t in range(5):
    b = arm(f"{LR}/multi_t{t}/base/off{{o}}_r{{a}}/result.json", 50, 2)
    ae = arm(f"{LR}/multi_t{t}/aegis/off{{o}}_r{{a}}/result.json", 50, 2)
    fb = arm(f"{LR}/multi_t{t}/nocurr/off{{o}}_r{{a}}/result.json", 50, 2)
    tasks.append((f"L1 T{t}", b, ae, fb))
for tag, name in (("apple", "L2 T0 apple"), ("lemon", "L2 T1 lemon"), ("mango", "L2 T2 mango"), ("onion", "L2 T3 onion"), ("tomato", "L2 T4 tomato")):
    b, ae, fb = L2[tag]; tasks.append((name, b, ae, fb))

print("\n### Score B = exp(-½[(1-SR)/(1-SR_base) + pCC/pCC_base])，bootstrap 2000 次（按 offset 重抽，三臂同一批 offset）\n")
print("| 任务 | B base | B AEGIS [95% CI] | B FailBank [95% CI] | P(B_FB > B_AEGIS) | 有效重抽 |\n|---|---|---|---|---|---|")
for name, b, ae, fb in tasks:
    offs = sorted(set(b) & set(ae) & set(fb))
    flat = lambda D, os_: [c for o in os_ for c in D[o]]
    pa = B(flat(ae, offs), flat(b, offs)); pf = B(flat(fb, offs), flat(b, offs))
    ba, bf, wins, valid = [], [], 0, 0
    for _ in range(2000):
        s = [random.choice(offs) for _ in offs]
        x = B(flat(ae, s), flat(b, s)); y = B(flat(fb, s), flat(b, s))
        if x is None or y is None:
            continue
        valid += 1; ba.append(x); bf.append(y); wins += y > x
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))] if v else float("nan")
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
    print(f"| {name} | 0.368 | {fmt(pa)} [{q(ba,.025):.3f}, {q(ba,.975):.3f}] | {fmt(pf)} [{q(bf,.025):.3f}, {q(bf,.975):.3f}] | {wins/max(valid,1):.3f} | {valid} |")
