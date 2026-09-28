#!/usr/bin/env python3
"""B3 (review 2026-09-15): how many training verdicts would flip under other guard thresholds.
Sources: every metrics.json / metrics_rejected.json under lora_dagger (bounded globs), every
VALIDATION_POINT line in se_vla/joblogs, and rejected-run ratios recorded in RESULTS_FOR_PAPER
(for runs trained before VALIDATION_POINT was logged). stdlib only."""
import glob, json, os, re, statistics, subprocess
LR = "${WORK_ROOT}/lora_dagger"
J = "${SE_VLA_ROOT}/joblogs"
runs = {}  # key -> dict(ratio, drift, step, src, accepted_at_110_005)

def add(key, ratio, drift, step, src):
    if ratio is None:
        return
    old = runs.get(key)
    if old is None or (step or 0) >= (old["step"] or 0):
        runs[key] = dict(ratio=float(ratio), drift=None if drift is None else float(drift), step=step, src=src)

pats = [f"{LR}/*/adapter/*/offset_0/metrics*.json", f"{LR}/*/*/adapter/*/offset_0/metrics*.json",
        f"{LR}/*/*/*/adapter/*/offset_0/metrics*.json", f"{LR}/*/offset_0/metrics*.json",
        f"{LR}/*/*/offset_0/metrics*.json"]
seen = set()
for pat in pats:
    for f in glob.glob(pat):
        if f in seen:
            continue
        seen.add(f)
        try:
            m = json.load(open(f))
        except Exception:
            continue
        key = os.path.dirname(os.path.dirname(f)).replace(LR + "/", "")
        add(key, m.get("best_quiet_flow_ratio"), m.get("best_quiet_action_drift"), m.get("best_step"), "metrics")

vp = re.compile(r"VALIDATION_POINT step=(\d+) triggered_loss=[0-9.eE+-]+ quiet_flow_ratio=([0-9.]+) quiet_drift=([0-9.]+)")
logs = subprocess.run(["grep", "-l", "VALIDATION_POINT", "-r", J], capture_output=True, text=True).stdout.split()
for f in logs:
    txt = open(f, errors="ignore").read()
    pts = vp.findall(txt)
    if not pts:
        continue
    arm = re.findall(r"arm=(\S+)", txt)
    final = re.findall(r"final=(\S+)|GUARD_REJECT_METRICS (\S+)", txt)
    key = "log:" + os.path.basename(f) + (":" + arm[0] if arm else "")
    step, r, d = max(pts, key=lambda x: int(x[0]))
    add(key, r, d, int(step), "joblog")

# Rejected runs trained before VALIDATION_POINT existed; values quoted from RESULTS_FOR_PAPER.
known = {
    "results:TAU02 (se_tau, qw0, dynamic)": (1.2308, 0.0075, "RESULTS 3-bis, Tau02Re.o1396131"),
    "results:9.9k bank 800 steps qw0 (dynamic)": (1.1016, 0.0094, "RESULTS 4 RQ3 table"),
    "results:9.9k bank 2400 steps qw0 (dynamic)": (1.1813, 0.0104, "RESULTS 4 RQ3 table"),
    "results:pi0 k=1 qw0 (se_pi0)": (1.477, None, "RESULTS 6b"),
}
for k, (r, d, s) in known.items():
    if not any(abs(v["ratio"] - r) < 5e-4 for v in runs.values()):
        runs[k] = dict(ratio=r, drift=d, step=None, src=s)

def acc(v, tl, td):
    return v["ratio"] <= tl and (v["drift"] is None or v["drift"] <= td)

out = []
rs = sorted(v["ratio"] for v in runs.values())
ds = sorted(v["drift"] for v in runs.values() if v["drift"] is not None)
out.append("# B3 训练闸门敏感性（自动生成）\n")
out.append(f"run 总数 {len(runs)}（metrics {sum(v['src']=='metrics' for v in runs.values())}，日志 {sum(v['src']=='joblog' for v in runs.values())}，RESULTS 记录 {sum(v['src'] not in ('metrics','joblog') for v in runs.values())}）\n")
q = lambda a, p: a[min(len(a) - 1, int(p * (len(a) - 1)))]
out.append(f"flow 比值：min {rs[0]:.4f} / p10 {q(rs,.1):.4f} / median {statistics.median(rs):.4f} / p90 {q(rs,.9):.4f} / max {rs[-1]:.4f}\n")
out.append(f"drift：min {ds[0]:.5f} / median {statistics.median(ds):.5f} / p90 {q(ds,.9):.5f} / max {ds[-1]:.5f}；超过 0.05 的 run 数 {sum(x>0.05 for x in ds)}\n")
out.append("\n## flow 比值在 1.00–1.30 的 run\n\n| run | flow 比值 | drift | 步数 | 来源 |\n|---|---|---|---|---|")
for k, v in sorted(runs.items(), key=lambda kv: kv[1]["ratio"]):
    if 1.0 <= v["ratio"] <= 1.30 or v["ratio"] > 1.30:
        dtxt = "" if v["drift"] is None else "%.5f" % v["drift"]
        out.append("| %s | %.4f | %s | %s | %s |" % (k, v["ratio"], dtxt, v["step"], v["src"]))
out.append("\n## 阈值网格：接受数，以及相对 (1.10, 0.05) 翻转的 run\n")
base = {k: acc(v, 1.10, 0.05) for k, v in runs.items()}
out.append("| τ_loss \\ τ_drift | " + " | ".join(str(td) for td in (0.01, 0.02, 0.05, 0.10)) + " |")
out.append("|---|---|---|---|---|")
flips = {}
for tl in (1.05, 1.10, 1.15, 1.20, 1.30):
    cells = []
    for td in (0.01, 0.02, 0.05, 0.10):
        a = {k: acc(v, tl, td) for k, v in runs.items()}
        f = sorted(k for k in runs if a[k] != base[k])
        flips[(tl, td)] = f
        cells.append(f"{sum(a.values())} 接受，{len(f)} 翻转")
    out.append(f"| {tl} | " + " | ".join(cells) + " |")
out.append("")
for (tl, td), f in flips.items():
    if f:
        items = []
        for k in f:
            dd = "" if runs[k]["drift"] is None else ", drift %.4f" % runs[k]["drift"]
            items.append("%s（%.4f%s）" % (k, runs[k]["ratio"], dd))
        out.append("- τ_loss=%s, τ_drift=%s 翻转：%s" % (tl, td, "；".join(items)))
dst = f"{LR}/review0913/lists/b3_guard_sensitivity.md"
open(dst, "w").write("\n".join(out) + "\n")
print("B3_DONE", dst, "runs", len(runs))
