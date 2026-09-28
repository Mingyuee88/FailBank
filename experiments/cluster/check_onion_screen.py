#!/usr/bin/env python3
"""Select one Onion candidate on offsets 0--9 using matched existing baselines."""

from __future__ import annotations

import json
import pathlib
import sys


lr = pathlib.Path(sys.argv[1])
screen = lr / "motivation_search/onion_l2_screen"


def read_paths(paths: list[pathlib.Path]) -> tuple[int, float, float]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        if result.get("status") != "pass":
            continue
        metric = result.get("metric_decomposition") or {}
        rows.append((float(metric.get("official_sr", 0)), float(metric.get("official_cc", 0))))
    if len(rows) != 30:
        raise RuntimeError(f"expected 30 pass cells, found {len(rows)}")
    return len(rows), 100 * sum(x for x, _ in rows) / len(rows), sum(y for _, y in rows) / len(rows)


def baseline(arm: str) -> tuple[int, float, float]:
    paths = [lr / f"l2_onion/{arm}/off{off}_r{advance}/result.json" for off in range(10) for advance in range(3)]
    return read_paths(paths)


stats = {"base": baseline("base"), "aegis": baseline("aegis")}
for arm in ("q20", "q20_r3"):
    stats[arm] = read_paths(sorted((screen / arm).glob("off*_r*/result.json")))
for arm, (n, sr, cc) in stats.items():
    print(f"{arm:7s} n={n:2d} SR={sr:5.1f} CC={cc:7.2f}")

_, base_sr, base_cc = stats["base"]
_, aegis_sr, aegis_cc = stats["aegis"]
feasible = []
for arm in ("q20", "q20_r3"):
    _, sr, cc = stats[arm]
    criteria = {
        "sr_at_least_70": sr >= 70,
        "sr_gain_over_aegis_at_least_50pp": sr >= aegis_sr + 50,
        "cc_at_most_80pct_base": cc <= 0.80 * base_cc,
    }
    for name, passed in criteria.items():
        print(f"{arm}_{name}={passed}")
    if all(criteria.values()):
        feasible.append((sr, -cc, arm))
selected = max(feasible)[2] if feasible else None
print(f"SELECTED={selected}")
print(f"CONFIRM={selected is not None}")
raise SystemExit(0 if selected else 2)
