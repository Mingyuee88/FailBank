#!/usr/bin/env python3
"""When the last hazard-suite base cells land, recompute the base table and update REVIEW_RESULTS.
Runs as an SGE job held on the pi0.5 L1 r1 arrays, so no interactive session is needed."""
import sys, re, shutil
sys.argv = ["x"]
R = "${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913"
exec(open(R + "/analyze_review.py").read().split("# ------------------------------------------------------------------ E1")[0])
MD = "${PAPER_ROOT}/REVIEW_RESULTS_20260914.md"
NM = ["T0", "T1", "T2", "T3", "T4"]
lines, floors, total = [], 0, 0
for bb, pre, label in (("pi0.5", "", "π0.5"), ("pi0", "pi0_", "π0")):
    for lv, root, reps in (("L1", "l1", 2 if bb == "pi0.5" else 1), ("L2", "l2", 3)):
        cells = []
        for t in range(5):
            A = arm("%s/e14/%s%s_t%d/base/off{o}_r{a}/result.json" % (O, pre, root, t), 50, reps)
            s = summ(A)
            cells.append("%s %.1f / %.2f / %.1f" % (NM[t], s["sr"], s["occ"], s["pcc"]))
            total += s["n"]; floors += s["sr"] <= 5.0
        lines.append("  - %s %s：%s" % (label, lv, "，".join(cells)))
block = ("- **base 已完成（全部 20 个任务臂，%d 个配对 offset）**，逐任务 SR / official CC / policy CC：\n" % total
         + "\n".join(lines)
         + "\n  - **余量**：20 个任务臂中 %d 个 base SR ≤ 5%%（地板）；按 E14 预注册的余量门，只有余量任务可做方法比较。\n" % floors)
m = open(MD).read()
i = m.index("- **base 已完成")
j = m.index("- **AEGIS 未做全量评测")
shutil.copy(MD, MD + ".bak.pre_e14_final")
m = m[:i] + block + m[j:]
m = m.replace("**base 与 AEGIS 两臂（09-18 收尾）**", "**base 与 AEGIS 两臂（09-18 收尾，base 已全部完成）**")
open(MD, "w").write(m)
print("E14_FINAL_MD_UPDATED cells=%d floor_arms=%d" % (total, floors))
