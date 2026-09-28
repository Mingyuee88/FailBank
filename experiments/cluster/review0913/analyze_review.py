#!/usr/bin/env python3
"""Review 2026-09-13: paired readouts for every finished review experiment.

Protocol (same as RESULTS §18-24): average sampler repeats within each offset, then a
two-sided exact sign test over offsets; ties within 1e-10 are dropped. Only cells with
status "pass" count, and a pair is formed only for offsets complete in both arms.
Each review cell is also checked for host and checkpoint provenance.
"""
import glob, json, math, os, re, statistics, sys
from collections import defaultdict

LR = "${WORK_ROOT}/lora_dagger"
O = f"{LR}/review0913"
TOL = 1e-10


def load(path):
    try:
        r = json.load(open(path))
    except Exception:
        return None
    if r.get("status") != "pass":
        return None
    md = r.get("metric_decomposition") or {}
    return dict(s=float(r.get("successes") or 0), pcc=md.get("policy_induced_cc"),
                occ=md.get("official_cc"), r=r)


def arm(pattern, noff, reps):
    """pattern has {o} and optionally {a}; returns {offset: [cells]} only for complete offsets."""
    out = {}
    for o in range(noff):
        cells = []
        for a in range(reps):
            c = load(pattern.format(o=o, a=a))
            if c is None:
                break
            cells.append(c)
        if len(cells) == reps:
            out[o] = cells
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def summ(A):
    cells = [c for v in A.values() for c in v]
    return dict(n=len(A), sr=100 * mean([c["s"] for c in cells]),
                pcc=mean([c["pcc"] for c in cells]), occ=mean([c["occ"] for c in cells]))


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return min(1.0, p)


def paired(A, B, key="s", higher_better=True):
    offs = sorted(set(A) & set(B))
    w = l = t = 0
    for o in offs:
        a = mean([c[key] for c in A[o]]); b = mean([c[key] for c in B[o]])
        d = a - b if higher_better else b - a
        if abs(d) <= TOL:
            t += 1
        elif d > 0:
            w += 1
        else:
            l += 1
    return w, l, t, sign_p(w, l), len(offs)


def row(name, A, B, bname):
    if not A or not B:
        return f"| {name} vs {bname} | — | — | — | — | — |"
    offs = sorted(set(A) & set(B))
    As = summ({o: A[o] for o in offs}); Bs = summ({o: B[o] for o in offs})
    w, l, t, p, n = paired(A, B, "s")
    cw, cl, ct, cp, _ = paired(A, B, "pcc", higher_better=False)
    return (f"| {name} vs {bname} | {n} | {As['sr']:.1f} vs {Bs['sr']:.1f} | {w}/{l}/{t} p={p:.4f} | "
            f"{As['pcc']:.2f} vs {Bs['pcc']:.2f} | {cw}/{cl}/{ct} p={cp:.4f} |")


HDR = ("| 比较 | 配对 offset | SR（%） | SR 胜/负/平 | policy CC | CC 胜/负/平（低为胜） |\n"
       "|---|---|---|---|---|---|")


def provenance(root, expect_host, expect_ck):
    """Count cells whose host.txt or restored checkpoint differ from what was intended."""
    bad_host = bad_ck = n = 0
    for d in glob.glob(f"{root}/*"):
        if not os.path.exists(f"{d}/result.json"):
            continue
        n += 1
        h = open(f"{d}/host.txt").read().strip() if os.path.exists(f"{d}/host.txt") else ""
        if not h.startswith(expect_host):
            bad_host += 1
        if expect_ck is None:
            continue
        txt = ""
        for f in glob.glob(f"{d}/server_port*.log"):
            txt += open(f, errors="ignore").read()
        m = re.findall(r"Finished restoring checkpoint.* from (\S+)", txt)
        if not m or not m[-1].rstrip(".").startswith(expect_ck.rstrip("/")):
            bad_ck += 1
    return n, bad_host, bad_ck


def section(title):
    print(f"\n## {title}\n")


BASE05 = "${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned"
BASE0 = "${SE_VLA_ROOT}/checkpoints/pi0_vla_arena_finetuned"
prov = []

# ------------------------------------------------------------------ E1
section("E1 归因对照")
ref = {a: arm(f"{LR}/l2_apple/{a}/off{{o}}_r{{a}}/result.json", 50, 3) for a in ("base", "ncs1", "ncs2", "ncs3")}
print("### L2 apple（50 offsets × 3 条件，HOST_A）\n")
print(HDR)
for s in (1, 2, 3):
    fb = ref[f"ncs{s}"]
    print(row(f"FailBank ncs{s}", fb, ref["base"], "base"))
    for c in ("sft0", "sham", "sftpos"):
        A = arm(f"{O}/e1/l2_apple/{c}_s{s}/off{{o}}_r{{a}}/result.json", 50, 3)
        prov.append((f"e1/l2_apple/{c}_s{s}",) + provenance(f"{O}/e1/l2_apple/{c}_s{s}", "HOST_A", f"{O}/e1/ckpt/{c}_s{s}"))
        print(row(f"{c}_s{s}", A, ref["base"], "base"))
        print(row(f"{c}_s{s}", A, fb, f"ncs{s}"))
for t, host in ((1, "HOST_C"), (4, "HOST_B")):
    print(f"\n### L1 T{t}（50 offsets × 2 条件，{host}，seed 1）\n")
    print(HDR)
    b = arm(f"{LR}/multi_t{t}/base/off{{o}}_r{{a}}/result.json", 50, 2)
    fb = arm(f"{LR}/multi_t{t}/ncs1/off{{o}}_r{{a}}/result.json", 50, 2)
    print(row("FailBank ncs1", fb, b, "base"))
    for c in ("sft0", "sham", "sftpos"):
        A = arm(f"{O}/e1/l1_t{t}/{c}_s1/off{{o}}_r{{a}}/result.json", 50, 2)
        prov.append((f"e1/l1_t{t}/{c}_s1",) + provenance(f"{O}/e1/l1_t{t}/{c}_s1", host, None))
        print(row(f"{c}_s1", A, b, "base"))
        print(row(f"{c}_s1", A, fb, "ncs1"))

# ------------------------------------------------------------------ E2
section("E2 主 adapter 跑动态 L1-T3 / T4（18 offsets × 1 条件，HOST_D）")
pool = defaultdict(dict)
for t in (3, 4):
    print(f"\n### 动态 L1-T{t}\n")
    print(HDR)
    b = arm(f"{LR}/tau_eval_t{t}/base_off{{o}}/result.json", 18, 1)
    refs = {"tau00": arm(f"{LR}/tau_eval_t{t}/tau00_off{{o}}/result.json", 18, 1)}
    if t == 3:
        refs["localsh"] = arm(f"{LR}/tau_eval_t3/localsh_off{{o}}/result.json", 18, 1)
    for a in ("nocurr", "ncs1", "ncs2", "ncs3"):
        A = arm(f"{O}/e2/tau_eval_t{t}/{a}/off{{o}}_r0/result.json", 18, 1)
        prov.append((f"e2/tau_eval_t{t}/{a}",) + provenance(f"{O}/e2/tau_eval_t{t}/{a}", "HOST_D", None))
        print(row(a, A, b, "base"))
        for o, v in A.items():
            pool[a][(t, o)] = v
        if a == "nocurr":
            for rn, R in refs.items():
                print(row(a, A, R, rn))
    for rn, R in refs.items():
        print(row(rn, R, b, "base"))
    pool["base"].update({(t, o): v for o, v in b.items()})
print("\n### 两任务合并（n=36）\n")
print(HDR)
for a in ("nocurr", "ncs1", "ncs2", "ncs3"):
    print(row(a, pool[a], pool["base"], "base"))

# ------------------------------------------------------------------ E3
section("E3 静态 L1 T0 / T3（50 offsets × 2 条件）")
l1pool = defaultdict(dict)
for t, host in ((0, "HOST_B"), (3, "HOST_C")):
    print(f"\n### L1 T{t}（{host}）\n")
    print(HDR)
    b = arm(f"{LR}/multi_t{t}/base/off{{o}}_r{{a}}/result.json", 50, 2)
    arms = {a: arm(f"{LR}/multi_t{t}/{a}/off{{o}}_r{{a}}/result.json", 50, 2) for a in ("aegis", "nocurr", "ncs1", "ncs2", "ncs3", "r1b", "r2")}
    for a in ("aegis", "nocurr", "ncs1", "ncs2", "ncs3"):
        prov.append((f"multi_t{t}/{a}",) + provenance(f"{LR}/multi_t{t}/{a}", host, None))
    for a in ("nocurr", "ncs1", "ncs2", "ncs3", "aegis", "r1b", "r2"):
        print(row(a, arms[a], b, "base"))
    print(row("nocurr", arms["nocurr"], arms["aegis"], "aegis"))
    srs = [summ(arms[a])["sr"] for a in ("ncs1", "ncs2", "ncs3") if arms[a]]
    pccs = [summ(arms[a])["pcc"] for a in ("ncs1", "ncs2", "ncs3") if arms[a]]
    if len(srs) == 3:
        print(f"\nT{t} 三个训练顺序 ncs1–3：SR {statistics.mean(srs):.1f} ± {statistics.stdev(srs):.1f}，"
              f"policy CC {statistics.mean(pccs):.2f} ± {statistics.stdev(pccs):.2f}；base SR {summ(b)['sr']:.1f}，policy CC {summ(b)['pcc']:.2f}")
print("\n### L1 聚合（nocurr vs base，任务×offset 为单位）\n")
print(HDR)
for t in (0, 1, 2, 3, 4):
    b = arm(f"{LR}/multi_t{t}/base/off{{o}}_r{{a}}/result.json", 50, 2)
    a = arm(f"{LR}/multi_t{t}/nocurr/off{{o}}_r{{a}}/result.json", 50, 2)
    if not b or not a:
        print(f"| T{t} 无 multi_t{t} 数据（base {len(b)} / nocurr {len(a)} offsets） | | | | | |")
        continue
    print(row(f"T{t} nocurr", a, b, "base"))
    for o in set(a) & set(b):
        l1pool["nocurr_all"][(t, o)] = a[o]; l1pool["base_all"][(t, o)] = b[o]
        if t != 2:
            l1pool["nocurr_ho"][(t, o)] = a[o]; l1pool["base_ho"][(t, o)] = b[o]
print(row("未训练四任务 T0/T1/T3/T4 nocurr", l1pool["nocurr_ho"], l1pool["base_ho"], "base"))
print(row("五任务（含 in-sample T2）nocurr", l1pool["nocurr_all"], l1pool["base_all"], "base"))

# ------------------------------------------------------------------ E4
section("E4 打标签用的特权几何盾在环（静态 L2，50 offsets × 3 条件）")
print(HDR)
for task, sub, host in (("apple", "l2_apple", "HOST_A"), ("mango", "l2_full", "HOST_D"), ("onion", "l2_onion", "HOST_B")):
    A = arm(f"{O}/e4/{sub}/teacherloop/off{{o}}_r{{a}}/result.json", 50, 3)
    prov.append((f"e4/{sub}/teacherloop",) + provenance(f"{O}/e4/{sub}/teacherloop", host, BASE05))
    b = arm(f"{LR}/{sub}/base/off{{o}}_r{{a}}/result.json", 50, 3)
    ae = arm(f"{LR}/{sub}/aegis/off{{o}}_r{{a}}/result.json", 50, 3)
    fb = arm(f"{LR}/{sub}/ncs1/off{{o}}_r{{a}}/result.json", 50, 3)
    div = tot = 0
    for o in set(A) & set(b):
        for x, y in zip(A[o], b[o]):
            tot += 1
            div += (x["s"] != y["s"]) or (x["occ"] != y["occ"]) or (x["pcc"] != y["pcc"])
    print(row(f"{task} teacherloop", A, b, "base"))
    print(row(f"{task} teacherloop", A, ae, "aegis"))
    print(row(f"{task} aegis", ae, b, "base"))
    print(row(f"{task} FailBank ncs1", fb, b, "base"))
    print(f"| {task} 有效性：与同主机 base 逐格分歧 {div}/{tot} = {100 * div / max(tot, 1):.0f}% | | | | | |")

# ------------------------------------------------------------------ E5/E6
section("E5 / E6 bank 构建消融（λ_q=0、800 步、seed 1）")
print("训练闸门：`oo_r1` 通过（flow 0.972），`accum_matched` 通过（flow 1.008）；**`il_r1`、`r2only` 被拒，无 adapter**。\n")
for sub, host, noff, reps, refroot in (("l2_apple", "HOST_A", 50, 3, f"{LR}/l2_apple"),
                                       ("l1_t1", "HOST_C", 50, 2, f"{LR}/multi_t1"),
                                       ("l1_t4", "HOST_B", 50, 2, f"{LR}/multi_t4")):
    print(f"\n### {sub}\n")
    print(HDR)
    b = arm(f"{refroot}/base/off{{o}}_r{{a}}/result.json", noff, reps)
    fb = arm(f"{refroot}/ncs1/off{{o}}_r{{a}}/result.json", noff, reps)
    ar = {}
    for a, e in (("oo_r1", "e5"), ("accum_matched", "e6")):
        ar[a] = arm(f"{O}/e56/{sub}/{a}/off{{o}}_r{{a}}/result.json", noff, reps)
        prov.append((f"e56/{sub}/{a}",) + provenance(f"{O}/e56/{sub}/{a}", host, None))
        print(row(a, ar[a], b, "base"))
        print(row(a, ar[a], fb, "ncs1（完整累积 bank）"))
    print(row("accum_matched", ar["accum_matched"], ar["oo_r1"], "oo_r1"))
    # Last-step diagnostics of the guard-rejected arms (guard limits 1e9; not deployable).
    for a, e in (("il_r1_diag", "e5"), ("r2only_diag", "e6")):
        ar[a] = arm(f"{O}/e56/{sub}/{a}/off{{o}}_r{{a}}/result.json", noff, reps)
        if not ar[a]:
            continue
        prov.append((f"e56/{sub}/{a}",) + provenance(f"{O}/e56/{sub}/{a}", host, f"{O}/{e}/ckpt/{a}"))
        print(row(f"{a}（诊断）", ar[a], b, "base"))
        print(row(f"{a}（诊断）", ar[a], fb, "ncs1（完整累积 bank）"))
    if ar.get("il_r1_diag"):
        print(row("il_r1_diag（诊断）", ar["il_r1_diag"], ar["oo_r1"], "oo_r1"))
    if ar.get("r2only_diag"):
        print(row("accum_matched", ar["accum_matched"], ar["r2only_diag"], "r2only_diag（诊断）"))
    print(f"\n完成 offset：oo_r1 {len(ar['oo_r1'])}/{noff}，accum_matched {len(ar['accum_matched'])}/{noff}，"
          f"il_r1_diag {len(ar.get('il_r1_diag') or {})}/{noff}，r2only_diag {len(ar.get('r2only_diag') or {})}/{noff}")

# ------------------------------------------------------------------ E7
section("E7 部署开销（L2 apple 前 10 offsets，单并发，HOST_A）")
print("| 臂 | 格数 | 总墙钟中位（秒） | 执行步数中位 | 墙钟/步中位（秒） | SR（%） |\n|---|---|---|---|---|---|")
for a in ("base", "failbank_ncs1", "aegis"):
    walls, steps, per, succ = [], [], [], []
    for d in sorted(glob.glob(f"{O}/e7/l2_apple/{a}/off*_r0")):
        c = load(f"{d}/result.json")
        if c is None or not os.path.exists(f"{d}/timing.txt"):
            continue
        t0, t1 = map(float, open(f"{d}/timing.txt").read().split()[:2])
        txt = "\n".join(c["r"].get("log_tail") or [])
        for ep in c["r"].get("episodes") or []:
            if isinstance(ep, dict):
                txt += " " + json.dumps(ep)
        m = re.findall(r"after (\d+) timesteps", txt) or re.findall(r'"(?:steps|num_steps|timesteps)": (\d+)', txt)
        n = int(m[-1]) if m else None
        walls.append(t1 - t0); succ.append(c["s"])
        if n:
            steps.append(n); per.append((t1 - t0) / n)
    med = lambda xs: statistics.median(xs) if xs else float("nan")
    print(f"| {a} | {len(walls)} | {med(walls):.1f} | {med(steps):.0f} | {med(per):.3f} | {100 * mean(succ):.0f} |")

# ------------------------------------------------------------------ E8
section("E8 π0 监督前 k 步（λ_q=0；π0 L1，50 offsets × 1 条件，HOST_A）——进行中")
print(HDR)
for t in range(5):
    b = arm(f"{LR}/pi0_eval2/base_t{t}_off{{o}}/result.json", 50, 1)
    q05 = arm(f"{LR}/pi0_eval2/ours_t{t}_off{{o}}/result.json", 50, 1)
    for k in (10, 50):
        A = arm(f"{O}/e8/pi0_l1/pi0_k{k}_t{t}_off{{o}}/result.json", 50, 1)
        if not A:
            continue
        print(row(f"T{t} k={k}", A, b, "π0 base"))
        if q05:
            print(row(f"T{t} k={k}", A, q05, "k=1 λ_q=0.5"))
done = len(glob.glob(f"{O}/e8/pi0_l1/*/result.json"))
prov.append(("e8/pi0_l1 (all)",) + provenance(f"{O}/e8/pi0_l1", "HOST_A", None))
print(f"\n已完成 {done}/500 格。")

# ------------------------------------------------------------------ E9
section("E9 动态第一轮 700 步（动态 L2-T0，50 offsets × 1 条件，HOST_B）")
print(HDR)
A = arm(f"{O}/e9/dyn_l2_t0/dyn_r1_700/off{{o}}_r0/result.json", 50, 1)
prov.append(("e9/dyn_l2_t0/dyn_r1_700",) + provenance(f"{O}/e9/dyn_l2_t0/dyn_r1_700", "HOST_B", f"{O}/e9/ckpt/dyn_r1_700"))
refs = {a: arm(f"{LR}/dyn_r3_t0/{a}/off{{o}}/result.json", 50, 1) for a in ("base", "r1", "r2", "r3")}
for a in ("base", "r1", "r2", "r3"):
    print(row("dyn_r1_700", A, refs[a], a))
for a in ("r1", "r2", "r3"):
    print(row(a, refs[a], refs["base"], "base"))

# ------------------------------------------------------------------ provenance
section("来源校验")
print("| 目录 | 格数 | 主机不符 | checkpoint 不符（未查记为 —） |\n|---|---|---|---|")
for name, n, bh, bc in prov:
    print(f"| {name} | {n} | {bh} | {bc} |")
