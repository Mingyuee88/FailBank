#!/usr/bin/env python3
"""Aggregate the development screen and emit a preregistered confirmation decision."""

from __future__ import annotations

import json
import pathlib
import sys


def summarize(root: pathlib.Path, arm: str) -> tuple[int, float, float]:
    rows = []
    for path in sorted((root / arm).glob("off*_r*/result.json")):
        data = json.loads(path.read_text())
        if data.get("status") != "pass":
            continue
        metric = data.get("metric_decomposition") or {}
        rows.append((float(metric.get("official_sr", 0.0)), float(metric.get("official_cc", 0.0))))
    if len(rows) != 30:
        raise RuntimeError(f"{arm}: expected 30 pass cells, found {len(rows)}")
    return len(rows), 100.0 * sum(x[0] for x in rows) / len(rows), sum(x[1] for x in rows) / len(rows)


root = pathlib.Path(sys.argv[1])
stats = {arm: summarize(root, arm) for arm in ("base", "aegis", "q20", "q20_r3")}
for arm, (n, sr, cc) in stats.items():
    print(f"{arm:5s} n={n:2d} SR={sr:5.1f} CC={cc:7.2f}")

_, base_sr, base_cc = stats["base"]
_, aegis_sr, aegis_cc = stats["aegis"]
def gate(arm: str) -> dict[str, bool]:
    _, ours_sr, ours_cc = stats[arm]
    return {
        "sr_at_least_30": ours_sr >= 30.0,
        "sr_gain_at_least_20pp": ours_sr >= max(base_sr, aegis_sr) + 20.0,
        "cc_at_most_80pct_base": ours_cc <= 0.80 * base_cc,
        "cc_within_25pct_aegis": ours_cc <= 1.25 * aegis_cc,
    }

# Fixed before observing either FailBank candidate on Lemon. The screen offsets
# only select R2 versus R3 and are excluded from all paper figures. Among feasible
# candidates, maximize SR (the stated primary target), breaking an exact tie by CC.
feasible = []
for arm in ("q20", "q20_r3"):
    criteria = gate(arm)
    for name, passed in criteria.items():
        print(f"{arm}_{name}={passed}")
    if all(criteria.values()):
        _, sr, cc = stats[arm]
        feasible.append((sr, -cc, arm))
selected = max(feasible)[2] if feasible else None
decision = selected is not None
print(f"SELECTED={selected}")
print(f"CONFIRM={decision}")
raise SystemExit(0 if decision else 2)
