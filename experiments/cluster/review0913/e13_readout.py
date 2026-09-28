import sys, math, glob, os, json, random
sys.argv = ["x"]; random.seed(0)
exec(open("analyze_review.py").read().split("# ------------------------------------------------------------------ E1")[0])
NM = ["Apple", "Lemon", "Mango", "Onion", "Tomato"]; TAG = [n.lower() for n in NM]
H2 = dict(apple="HOST_A", lemon="HOST_A", mango="HOST_D", onion="HOST_D", tomato="HOST_A")
CK2 = f"{O}/e13/ckpt/pi0_q05_s2/offset_0"

def prov(root, host):
    ds = [d for d in glob.glob(root + "/*") if os.path.exists(d + "/result.json")]
    bh = bc = bf = 0
    for d in ds:
        if not open(d + "/host.txt").read().strip().startswith(host): bh += 1
        txt = "".join(open(f, errors="ignore").read() for f in glob.glob(d + "/server_port*.log"))
        rs = [l for l in txt.splitlines() if "Finished restoring checkpoint" in l]
        if not rs or not all(CK2 in l for l in rs): bc += 1
        if (json.load(open(d + "/result.json")).get("architecture") or {}).get("architecture_family") != "pi0": bf += 1
    return f"{len(ds)} 格，主机不符 {bh}，checkpoint 不符 {bc}，family≠pi0 {bf}"

def avg(arms): return {o: [c for A in arms for c in A[o]] for o in range(50) if all(o in A for A in arms)}
def S(A): s = summ(A); return s["sr"], s["occ"], s["pcc"]
def B(sr, pcc, bsr, bpcc):
    if bpcc <= 0 or bsr >= 100: return None
    return math.exp(-0.5 * ((1 - sr / 100) / (1 - bsr / 100) + pcc / bpcc))
fmtB = lambda v: "—" if v is None else f"{v:.3f}"

print("# E13 读数：π0 FailBank 三个训练顺序\n")
print("## 闸门\n- seed 0（PI0_Q05，已有）：flow 1.0829，drift 0.0122，通过\n- seed 1（pi0_q05_s1）：flow **1.2214**，drift 0.0089，**被拒**（按预注册不重训、不放宽）\n- seed 2（pi0_q05_s2）：flow 1.0171，drift 0.0091，通过\n\nπ0 FailBank 均值 = seed 0 与 seed 2 两个通过的 adapter（2 of 3 passed）。\n")
print("## 来源校验（seed 2）\n")
for t in range(5): print(f"- L1 T{t}：" + prov(f"{O}/e13/pi0_l1_t{t}/pi0_q05_s2", "HOST_A"))
for g in TAG: print(f"- L2 {g}：" + prov(f"{O}/e13/pi0_l2_{g}/pi0_q05_s2", H2[g]))
hc = open(f"{O}/lists/e13_hostcheck.txt").read().strip().splitlines()[-2:]
print(f"- apple / lemon 移到 HOST_A：同型号跨主机验证 {hc[0]}，{hc[1]}（与 l40s-005 上 E11 原格逐格一致，所以仍与 005 上的 base / AEGIS / seed 0 配对）\n")

R = []
for t in range(5):
    b = arm(f"{LR}/pi0_eval2/base_t{t}_off{{o}}/result.json", 50, 1)
    a = arm(f"{O}/e11/pi0_l1_t{t}/aegis/off{{o}}_r{{a}}/result.json", 50, 1)
    s0 = arm(f"{LR}/pi0_eval2/ours_t{t}_off{{o}}/result.json", 50, 1)
    s2 = arm(f"{O}/e13/pi0_l1_t{t}/pi0_q05_s2/off{{o}}_r{{a}}/result.json", 50, 1)
    R.append(("L1", NM[t], b, a, s0, s2))
for t, g in enumerate(TAG):
    e = f"{O}/e11/pi0_l2_{g}"
    R.append(("L2", NM[t], arm(f"{e}/base/off{{o}}_r{{a}}/result.json", 50, 3), arm(f"{e}/aegis/off{{o}}_r{{a}}/result.json", 50, 3),
              arm(f"{e}/failbank/off{{o}}_r{{a}}/result.json", 50, 3), arm(f"{O}/e13/pi0_l2_{g}/pi0_q05_s2/off{{o}}_r{{a}}/result.json", 50, 3)))

print("## 逐任务（SR / CC / CCπ / B）\n")
print("| 级别 | 任务 | offsets（base/AE/s0/s2） | Base | AEGIS | seed 0 | seed 2 | **FailBank 均值（s0,s2）** | best-of-2（按 SR，只作描述） |")
print("|---|---|---|---|---|---|---|---|---|")
agg = {"base": [], "aegis": [], "mean": [], "best": [], "s0": [], "s2": []}
for lv, nm, b, a, s0, s2 in R:
    m = avg([s0, s2]); sb = S(b)
    cols = {}
    for k, X in (("base", b), ("aegis", a), ("s0", s0), ("s2", s2), ("mean", m)):
        v = S(X); cols[k] = v; agg[k].append(v)
    best = max((cols["s0"], cols["s2"]), key=lambda v: (v[0], -v[2])); agg["best"].append(best)
    f = lambda v: f"{v[0]:.1f} / {v[1]:.2f} / {v[2]:.2f} / {fmtB(B(v[0], v[2], sb[0], sb[2]))}"
    print(f"| {lv} | {nm} | {len(b)}/{len(a)}/{len(s0)}/{len(s2)} | {f(cols['base'])} | {f(cols['aegis'])} | {f(cols['s0'])} | {f(cols['s2'])} | **{f(cols['mean'])}** | {f(best)} |")

print("\n## 10 任务等权平均与汇总 B\n")
print("| 臂 | SR | CC | CCπ | 汇总 B |\n|---|---|---|---|---|")
mb = [sum(v[i] for v in agg["base"]) / 10 for i in range(3)]
for k, name in (("base", "Base"), ("aegis", "AEGIS"), ("s0", "FailBank seed 0（原主表）"), ("s2", "FailBank seed 2"), ("mean", "**FailBank 均值**"), ("best", "FailBank best-of-2（描述）")):
    m = [sum(v[i] for v in agg[k]) / 10 for i in range(3)]
    print(f"| {name} | {m[0]:.1f} | {m[1]:.2f} | {m[2]:.2f} | {B(m[0], m[2], mb[0], mb[2]):.3f} |")

print("\n## 配对检验（FailBank 均值；任务×offset）\n")
print(HDR)
P = {k: {} for k in ("b", "a", "m", "s2")}
for lv, nm, b, a, s0, s2 in R:
    m = avg([s0, s2])
    print(row(f"{lv} {nm} FailBank 均值", m, b, "base"))
    print(row(f"{lv} {nm} FailBank 均值", m, a, "AEGIS"))
    print(row(f"{lv} {nm} seed 2", s2, b, "base"))
    for k, X in (("b", b), ("a", a), ("m", m), ("s2", s2)):
        for o, v in X.items(): P[k][(lv, nm, o)] = v
for lab, sel in (("L1 五任务", lambda k: k[0] == "L1"), ("L1 未训练四任务", lambda k: k[0] == "L1" and k[1] != "Mango"),
                 ("L2 五任务", lambda k: k[0] == "L2"), ("10 任务", lambda k: True)):
    Q = {k: {kk: v for kk, v in P[k].items() if sel(kk)} for k in P}
    print(row(f"{lab} FailBank 均值", Q["m"], Q["b"], "base"))
    print(row(f"{lab} FailBank 均值", Q["m"], Q["a"], "AEGIS"))
    print(row(f"{lab} seed 2", Q["s2"], Q["b"], "base"))
