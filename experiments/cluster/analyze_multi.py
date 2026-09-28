#!/usr/bin/env python3
"""Analyze multi-arm sweeps. Aggregates repeats per offset BEFORE testing:
r0/r1 are two sampler-advance conditions of the SAME offset and are not
independent -- treating them as separate units inflates n and fakes power.

Layouts handled:
  multi_<tag>/<arm>/off<N>_r<A>/result.json      (run_multitask.sh)
  pi0_eval2/<arm>_t<T>_off<N>/result.json        (run_pi0_eval2.sh)
"""
import json, os, re, sys, glob, collections
from math import comb

def sign_test(a, b):
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(a, b) + 1)) / 2 ** n)

def read_cell(path):
    try:
        r = json.load(open(path))
    except Exception:
        return None
    if r.get("status") != "pass":
        return None
    su = r.get("successes")
    su = sum(su) if isinstance(su, list) else (su or 0)
    ep = r.get("episodes_completed") or r.get("episodes_requested") or 1
    md = r.get("metric_decomposition") or {}
    h = os.path.join(os.path.dirname(path), "host.txt")
    host = open(h).read().strip().split('.')[0] if os.path.exists(h) else "?"
    return dict(sr=su / max(ep, 1),
                occ=float(md.get("official_cc") or 0),
                pcc=float(md.get("policy_induced_cc") or 0),
                host=host)

def collect(root):
    """-> {arm: {offset: [cell, ...]}}, hosts"""
    arms = collections.defaultdict(lambda: collections.defaultdict(list))
    hosts = set()
    for path in glob.glob(os.path.join(root, "*", "*", "result.json")):
        parts = path.split(os.sep)
        arm, cell = parts[-3], parts[-2]
        m = re.match(r"off(\d+)_r(\d+)$", cell)
        if not m:
            continue
        c = read_cell(path)
        if c:
            arms[arm][int(m.group(1))].append(c)
            hosts.add(c["host"])
    for path in glob.glob(os.path.join(root, "*", "result.json")):
        cell = path.split(os.sep)[-2]
        m = re.match(r"(.+)_t(\d+)_off(\d+)$", cell)
        if not m:
            continue
        c = read_cell(path)
        if c:
            arms[f"{m.group(1)}_t{m.group(2)}"][int(m.group(3))].append(c)
            hosts.add(c["host"])
    return arms, hosts

def agg(cells, key):
    return sum(c[key] for c in cells) / len(cells)

def main(root, base_arm=None):
    arms, hosts = collect(root)
    if not arms:
        print(f"{root}: 无数据"); return
    print(f"\n=== {root} ===")
    tag = "OK-single-host" if len(hosts) == 1 else f"WARN-MIXED-HOST {sorted(hosts)}"
    print(f"hosts: {sorted(hosts)}  {tag}")
    if len(hosts) > 1:
        print("  !! 跨主机不可配对：GPU 型号可移动 SR 达 29 点。先隔离再分析。")
    rows = {}
    for arm in sorted(arms):
        offs = arms[arm]
        sr = 100 * sum(agg(v, "sr") for v in offs.values()) / len(offs)
        pcc = sum(agg(v, "pcc") for v in offs.values()) / len(offs)
        occ = sum(agg(v, "occ") for v in offs.values()) / len(offs)
        ncell = sum(len(v) for v in offs.values())
        rows[arm] = offs
        print(f"  {arm:14s} offsets={len(offs):3d} cells={ncell:4d} "
              f"SR={sr:5.1f}  polCC={pcc:7.3f}  offCC={occ:7.3f}")
    if base_arm is None:
        base_arm = "base" if "base" in rows else None
    if base_arm and base_arm in rows:
        print(f"  --- paired vs {base_arm} (unit = offset) ---")
        for arm in sorted(rows):
            if arm == base_arm:
                continue
            common = set(rows[arm]) & set(rows[base_arm])
            i = w = t = 0
            for o in common:
                x, y = agg(rows[arm][o], "sr"), agg(rows[base_arm][o], "sr")
                if x > y: i += 1
                elif x < y: w += 1
                else: t += 1
            if common:
                print(f"  {arm:14s} SR {i}/{w}/{t} over {len(common)} offsets  "
                      f"p={sign_test(i, w):.4f}")

if __name__ == "__main__":
    for r in sys.argv[1:] or ["."]:
        main(r)
