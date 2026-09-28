import sys
sys.argv = ["x"]
exec(open("analyze_review.py").read().split("# ------------------------------------------------------------------ E1")[0])


def pr(name, pat, noff, reps):
    A = arm(pat, noff, reps)
    if not A:
        print("| %s | 0 | | | |" % name)
        return
    s = summ(A)
    print("| %s | %d | %.1f | %.2f | %.2f |" % (name, s["n"], s["sr"], s["occ"], s["pcc"]))


print("| 臂 | offsets | SR | official CC | policy CC |\n|---|---|---|---|---|")
for t in (0, 1, 2, 3, 4):
    for a in ("base", "aegis", "nocurr", "ncs1", "ncs2", "ncs3"):
        pr("L1 T%d %s" % (t, a), LR + "/multi_t%d/%s/off{o}_r{a}/result.json" % (t, a), 50, 2)
for sub in ("l2_apple", "l2_full", "l2_onion"):
    pr(sub + " teacherloop", O + "/e4/%s/teacherloop/off{o}_r{a}/result.json" % sub, 50, 3)
    for a in ("base", "aegis", "ncs1"):
        pr("%s %s" % (sub, a), LR + "/%s/%s/off{o}_r{a}/result.json" % (sub, a), 50, 3)
