import sys
sys.argv = ["x"]
exec(open("analyze_review.py").read().split("# ------------------------------------------------------------------ E1")[0])
pool = defaultdict(dict)
print(HDR)
for t in range(5):
    b = arm(f"{LR}/pi0_eval2/base_t{t}_off{{o}}/result.json", 50, 1)
    q = arm(f"{LR}/pi0_eval2/ours_t{t}_off{{o}}/result.json", 50, 1)
    for k in (10, 50):
        A = arm(f"{O}/e8/pi0_l1/pi0_k{k}_t{t}_off{{o}}/result.json", 50, 1)
        print(row(f"T{t} k={k}", A, b, "π0 base"))
        print(row(f"T{t} k={k}", A, q, "k=1 λ_q=0.5"))
        for o, v in A.items():
            pool[f"k{k}"][(t, o)] = v
    print(row(f"T{t} k=1 λ_q=0.5", q, b, "π0 base"))
    for o, v in b.items():
        pool["base"][(t, o)] = v
    for o, v in q.items():
        pool["q05"][(t, o)] = v
for k in ("k10", "k50"):
    print(row(f"{k} 五任务合并", pool[k], pool["base"], "π0 base"))
    print(row(f"{k} 五任务合并", pool[k], pool["q05"], "k=1 λ_q=0.5"))
print(row("k10 五任务合并", pool["k10"], pool["k50"], "k50"))
print(row("k=1 λ_q=0.5 五任务合并", pool["q05"], pool["base"], "π0 base"))
for name, p in (("k10", pool["k10"]), ("k50", pool["k50"]), ("q05", pool["q05"]), ("base", pool["base"])):
    s = summ(p)
    print("SUMMARY", name, "n=%d sr=%.1f occ=%.2f pcc=%.2f" % (s["n"], s["sr"], s["occ"], s["pcc"]))
