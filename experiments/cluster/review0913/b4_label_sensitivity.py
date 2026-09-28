#!/usr/bin/env python3
"""B4 (review 2026-09-15): how much the main training bank changes when the record-admission
thresholds move. Recomputes, from the unfiltered bank_r2 records and the per-episode raw steps:
first risk step (cost_pair_min_distance < risk threshold), lead-time stage, keep_record, and
correction-vs-anchor targets exactly as build_round_records.py --mode s1s2 does; teacher buckets
are recomputed from the stored score and gate. The default configuration must reproduce the main
bank (6,535 train rows, 709 real corrections); any mismatch is reported, not hidden."""
import collections, importlib.util, json, math, sys
from pathlib import Path
LR = Path("${WORK_ROOT}/lora_dagger")
SRC = LR / "bank_r2"
MAIN = LR / "se_r2/s1s2_records/derived/folds/offset_0/train.jsonl"
L = Path("${SE_VLA_ROOT}/roundG/pi05_stage2/lora")
spec = importlib.util.spec_from_file_location("brr", L / "build_round_records.py")
brr = importlib.util.module_from_spec(spec); spec.loader.exec_module(brr)
DELTA = brr.DELTA_SETS["s1s2"]
out = []
def say(s=""):
    out.append(s); print(s, flush=True)

src_train = SRC / "derived/folds/offset_0/train.jsonl"
recs = [json.loads(l) for l in open(src_train)]
say(f"# B4 标签阈值敏感性（自动生成）\n\n源：`{src_train}`，{len(recs)} 条；DROP_STAGES={sorted(brr.DROP_STAGES)}；delta 阶段={sorted(DELTA)}\n")

# per-episode distance series
ep_dirs = {}
for ej in SRC.glob("episodes/*/*/*/episode.json"):
    ep_dirs[ej.parent.name] = ej.parent
for ej in SRC.glob("episodes/*/*/episode.json"):
    ep_dirs.setdefault(ej.parent.name, ej.parent)
need = {r["episode_id"] for r in recs}
dist = {}
missing = 0
for eid in need:
    d = ep_dirs.get(eid)
    if d is None or not (d / "raw_steps.jsonl").exists():
        missing += 1; continue
    series = []
    for line in open(d / "raw_steps.jsonl"):
        try:
            series.append((json.loads(line).get("runtime_info") or {}).get("cost_pair_min_distance"))
        except Exception:
            series.append(None)
    dist[eid] = series
say(f"回合：记录涉及 {len(need)} 个，找到逐步距离 {len(dist)} 个，缺失 {missing} 个\n")

def first_risk(eid, th):
    s = dist.get(eid)
    if s is None:
        return "MISSING"
    for i, d in enumerate(s):
        if d is not None and d < th:
            return i
    return None

def stage_for(lead, b_early, b_mid):
    if lead is None:
        return "NO_RISK"
    for name, lo, hi in (("S1_early", b_early, 10**9), ("S2_mid", b_mid, b_early), ("S3_emergency", 0, b_mid), ("POST_crossing", -10**9, 0)):
        if lo <= lead < hi:
            return name
    return "NO_RISK"

def bucket(score, gate, cuts):
    lo, mid, hi = cuts
    if not gate or score < lo: return "reject", 0.0
    if score < mid: return "low", 0.25
    if score < hi: return "med", 0.60
    return "high", 1.00

def build(th, b_early, b_mid, cuts):
    kept = {}
    for r in recs:
        eid = r["episode_id"]
        fr = first_risk(eid, th)
        if fr == "MISSING":
            stage = r.get("curriculum_stage")          # fall back to stored stage
        else:
            lead = None if fr is None else fr - int(r["step_index"])
            stage = stage_for(lead, b_early, b_mid)
        rr = dict(r); rr["curriculum_stage"] = stage
        if not brr.keep_record(rr):
            continue
        trig = bool((r.get("formal_projection") or {}).get("triggered"))
        t = r.get("teacher") or {}
        b, w = bucket(float(t.get("score", 0.0) or 0.0), bool(t.get("gate")), cuts) if trig else (None, 0.0)
        kind = "quiet"
        if trig:
            d = math.dist(brr.clip_nominal(r["nominal_action"]), r["executed_action"][:3])
            kind = "delta" if (stage in DELTA and d > 1e-9) else "anchor"
        kept[r["record_id"]] = dict(stage=stage, kind=kind, bucket=b, w=w)
    return kept

DEF = (0.1087, 30, 15, (0.25, 0.45, 0.70))
base = build(*DEF)
main_ids = {json.loads(l)["record_id"] for l in open(MAIN)}
nd = sum(v["kind"] == "delta" for v in base.values())
say(f"## 自检（默认阈值）\n\n重建训练集 {len(base)} 行（主 bank {len(main_ids)} 行），真实修正 {nd} 条（主 bank 709 条），"
    f"与主 bank 记录集合的对称差 {len(set(base) ^ main_ids)} 条\n")
stored_mismatch = sum(1 for r in recs if dist.get(r["episode_id"]) is not None and
                      (lambda fr: (None if fr is None else fr - int(r["step_index"])))(first_risk(r["episode_id"], 0.1087)) != r.get("steps_to_first_risk"))
say(f"默认阈值下重算的 steps_to_first_risk 与记录中存储值不一致的条数：{stored_mismatch}\n")

def compare(name, cfg):
    k = build(*cfg)
    added = set(k) - set(base); removed = set(base) - set(k); common = set(k) & set(base)
    kind_chg = sum(1 for i in common if k[i]["kind"] != base[i]["kind"])
    stage_chg = sum(1 for i in common if k[i]["stage"] != base[i]["stage"])
    b_chg = sum(1 for i in common if k[i]["bucket"] != base[i]["bucket"])
    mass_b = sum(v["w"] for v in base.values()); mass_k = sum(v["w"] for v in k.values())
    touched = len(added) + len(removed) + sum(1 for i in common if (k[i]["kind"], k[i]["bucket"]) != (base[i]["kind"], base[i]["bucket"]))
    nd_k = sum(v["kind"] == "delta" for v in k.values())
    return (f"| {name} | {len(k)} | {nd_k} | +{len(added)} / −{len(removed)} | {stage_chg} | {kind_chg} | {b_chg} | "
            f"{mass_k:.1f}（{100*(mass_k-mass_b)/mass_b:+.1f}%） | {touched}（{100*touched/len(base):.1f}%） |")

say("## 扰动结果（相对默认阈值）\n")
say("| 扰动 | 训练行 | 真实修正 | 新增 / 移除 | 阶段改变 | 修正↔锚定改变 | 教师档改变 | triggered 权重质量 | 受影响记录（占默认训练集） |")
say("|---|---|---|---|---|---|---|---|---|")
say(compare("默认 0.1087, 30/15, 0.25/0.45/0.70", DEF))
for f in (0.8, 1.2):
    say(compare(f"风险阈值 ×{f}（{0.1087*f:.4f}）", (0.1087 * f, 30, 15, DEF[3])))
for be, bm in ((24, 12), (36, 18)):
    say(compare(f"阶段边界 {be}/{bm}", (0.1087, be, bm, DEF[3])))
for s in (-0.05, 0.05):
    cuts = tuple(round(c + s, 2) for c in DEF[3])
    say(compare(f"教师分档 {cuts}", (0.1087, 30, 15, cuts)))
say("\n读法（预注册）：受影响记录 ≤10% → bank 构成对该阈值不敏感；>10% → 如实报告。")
dst = LR / "review0913/lists/b4_label_sensitivity.md"
dst.write_text("\n".join(out) + "\n")
print("B4_DONE", dst)
