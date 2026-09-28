import sys, random, math, glob, os, json, statistics
sys.argv = ["x"]
exec(open("analyze_review.py").read().split("# ------------------------------------------------------------------ E1")[0])
random.seed(0)
BASECK = "${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned"

def cells(root):
    return [d for d in glob.glob(root + "/*") if os.path.exists(d + "/result.json")]

def prov(root, expect_host, expect_ck=None, pi0=False, aegis=False):
    ds = cells(root); bad_h = bad_ck = bad_f = bad_p = 0
    for d in ds:
        h = open(d + "/host.txt").read().strip() if os.path.exists(d + "/host.txt") else ""
        if not h.startswith(expect_host): bad_h += 1
        if expect_ck:
            txt = ""
            for lf in glob.glob(d + "/server_port*.log"):
                txt += open(lf, errors="ignore").read()
            restores = [l for l in txt.splitlines() if "Finished restoring checkpoint" in l]
            if not restores or not all(expect_ck in l for l in restores): bad_ck += 1
        if pi0:
            r = json.load(open(d + "/result.json"))
            if (r.get("architecture") or {}).get("architecture_family") != "pi0": bad_f += 1
        if aegis:
            ok = None
            s = d + "/srd.jsonl"
            if os.path.exists(s):
                for line in open(s, errors="ignore"):
                    if "vlsa_audit" in line:
                        try: ok = json.loads(line).get("perception_ok")
                        except Exception: ok = False
                        break
            if ok is not True: bad_p += 1
    return f"{len(ds)} 格；主机不符 {bad_h}" + (f"；checkpoint 不符 {bad_ck}" if expect_ck else "") + (f"；family≠pi0 {bad_f}" if pi0 else "") + (f"；perception_ok≠true {bad_p}" if aegis else "")

def hosts(root):
    c = {}
    for d in cells(root):
        h = open(d + "/host.txt").read().strip() if os.path.exists(d + "/host.txt") else "?"
        c[h] = c.get(h, 0) + 1
    return c

def B(A, Bs):
    fa = 1 - mean([c["s"] for c in A]); fb = 1 - mean([c["s"] for c in Bs])
    pa = mean([c["pcc"] for c in A]); pb = mean([c["pcc"] for c in Bs])
    if fb <= 0 or pb <= 0: return None
    return math.exp(-0.5 * (fa / fb + pa / pb))

def bci(X, base):
    offs = sorted(set(X) & set(base)); flat = lambda D, s: [c for o in s for c in D[o]]
    pt = B(flat(X, offs), flat(base, offs))
    if pt is None: return "—"
    v = []
    for _ in range(2000):
        s = [random.choice(offs) for _ in offs]; x = B(flat(X, s), flat(base, s))
        if x is not None: v.append(x)
    v.sort(); return f"{pt:.3f} [{v[int(.025*(len(v)-1))]:.3f}, {v[int(.975*(len(v)-1))]:.3f}]"

def tab(name, arms, base):
    print(f"| {name} | " + " | ".join(f"{summ(A)['sr']:.1f} | {summ(A)['occ']:.2f} / {summ(A)['pcc']:.2f} | {bci(A, base) if A is not base else ('0.368' if B([c for v in base.values() for c in v],[c for v in base.values() for c in v]) else '—')}" for A in arms) + " |")

print("# E11 / E12 读数\n")
print("## E11 π0 补全\n")
print("### 来源校验\n")
PFB = f"{LR}/se_pi0_q05/ckpt_pi0fold_1405748/PI0_Q05"
for t in range(5):
    print("- L1 T%d AEGIS：" % t + prov(O + "/e11/pi0_l1_t%d/aegis" % t, "HOST_A", pi0=True, aegis=True))
pe = {}
for f in glob.glob(f"{LR}/pi0_eval2/*_off*"):
    h = open(f + "/host.txt").read().strip() if os.path.exists(f + "/host.txt") else "?"
    pe[h] = pe.get(h, 0) + 1
print(f"- pi0_eval2 主机分布：{pe}")
L2H = dict(apple="HOST_B", lemon="HOST_B", mango="HOST_D", onion="HOST_D", tomato="HOST_A")
for tag, h in L2H.items():
    R = f"{O}/e11/pi0_l2_{tag}"
    print(f"- L2 {tag}：base {prov(R+'/base', h, pi0=True)}｜FailBank {prov(R+'/failbank', h, PFB, pi0=True)}｜AEGIS {prov(R+'/aegis', h, pi0=True, aegis=True)}")

print("\n### π0 主表行（SR | official / policy CC | Score B [95% CI]）\n")
print("| 任务 | base SR | base CC | base B | AEGIS SR | AEGIS CC | AEGIS B | FailBank SR | FailBank CC | FailBank B |")
print("|---|---|---|---|---|---|---|---|---|---|")
T = ["T0 apple", "T1 lemon", "T2 mango", "T3 onion", "T4 tomato"]
L1 = {}; L2 = {}
for t in range(5):
    b = arm(f"{LR}/pi0_eval2/base_t{t}_off{{o}}/result.json", 50, 1)
    f = arm(f"{LR}/pi0_eval2/ours_t{t}_off{{o}}/result.json", 50, 1)
    a = arm(f"{O}/e11/pi0_l1_t{t}/aegis/off{{o}}_r{{a}}/result.json", 50, 1)
    L1[t] = (b, a, f); tab(f"L1 {T[t]}", [b, a, f], b)
for t, tag in enumerate(L2H):
    R = f"{O}/e11/pi0_l2_{tag}"
    b = arm(f"{R}/base/off{{o}}_r{{a}}/result.json", 50, 3)
    a = arm(f"{R}/aegis/off{{o}}_r{{a}}/result.json", 50, 3)
    f = arm(f"{R}/failbank/off{{o}}_r{{a}}/result.json", 50, 3)
    L2[t] = (b, a, f); tab(f"L2 {T[t]}", [b, a, f], b)

print("\n### π0 配对检验\n")
print(HDR)
for lvl, D in (("L1", L1), ("L2", L2)):
    for t in range(5):
        b, a, f = D[t]
        print(row(f"{lvl} {T[t]} FailBank", f, b, "base"))
        print(row(f"{lvl} {T[t]} AEGIS", a, b, "base"))
        print(row(f"{lvl} {T[t]} FailBank", f, a, "AEGIS"))
    P = {k: {} for k in "baf"}
    for t in range(5):
        for k, X in zip("baf", D[t]):
            for o, v in X.items(): P[k][(t, o)] = v
    print(row(f"{lvl} 五任务合并 FailBank", P["f"], P["b"], "base"))
    print(row(f"{lvl} 五任务合并 AEGIS", P["a"], P["b"], "base"))
    print(row(f"{lvl} 五任务合并 FailBank", P["f"], P["a"], "AEGIS"))
print("\n地板规则（π0 L2，base 与 FailBank SR 均 ≤5%）：" + ", ".join(f"{T[t]}={summ(L2[t][0])['sr']<=5 and summ(L2[t][2])['sr']<=5}" for t in range(5)))

print("\n## E12 flow 比值剂量-响应\n")
FR = {"s400": 1.00422, "ncs1 (800)": 1.00729, "s1600": 1.01645, "s2400": 1.03408, "s4800": 1.11030}
DR = {"s400": 0.008331, "ncs1 (800)": 0.009578, "s1600": 0.010817, "s2400": 0.012052, "s4800": 0.015018}
for n in (400, 1600, 2400, 4800):
    print(f"- s{n}：apple {prov(f'{O}/e12/l2_apple/s{n}', 'HOST_A', f'e12/ckpt/s{n}')}；L1 T4 {prov(f'{O}/e12/l1_t4/s{n}', 'HOST_B', f'e12/ckpt/s{n}')}")
print(f"- 参照主机：l2_apple base {hosts(f'{LR}/l2_apple/base')}，ncs1 {hosts(f'{LR}/l2_apple/ncs1')}；multi_t4 base {hosts(f'{LR}/multi_t4/base')}，ncs1 {hosts(f'{LR}/multi_t4/ncs1')}\n")
for task, sub, reps, E in (("L2 apple", "l2_apple", 3, "l2_apple"), ("L1 T4", "multi_t4", 2, "l1_t4")):
    b = arm(f"{LR}/{sub}/base/off{{o}}_r{{a}}/result.json", 50, reps)
    n8 = arm(f"{LR}/{sub}/ncs1/off{{o}}_r{{a}}/result.json", 50, reps)
    X = {"s400": arm(f"{O}/e12/{E}/s400/off{{o}}_r{{a}}/result.json", 50, reps), "ncs1 (800)": n8}
    for n in (1600, 2400, 4800):
        X[f"s{n}"] = arm(f"{O}/e12/{E}/s{n}/off{{o}}_r{{a}}/result.json", 50, reps)
    print(f"### {task}\n\n| 步数 | flow 比值 | drift | offsets | SR | official CC | policy CC |\n|---|---|---|---|---|---|---|")
    sb = summ(b); print(f"| base | — | — | {sb['n']} | {sb['sr']:.1f} | {sb['occ']:.2f} | {sb['pcc']:.2f} |")
    for k, A in X.items():
        s = summ(A); print(f"| {k} | {FR[k]:.4f} | {DR[k]:.4f} | {s['n']} | {s['sr']:.1f} | {s['occ']:.2f} | {s['pcc']:.2f} |")
    print("\n" + HDR)
    for k, A in X.items():
        print(row(k, A, b, "base"))
        if k != "ncs1 (800)": print(row(k, A, n8, "800 步"))
    print()
