#!/usr/bin/env python3
"""Select a Mango L1 candidate on offsets 0--9; never use them in the figure."""

from __future__ import annotations

import json
import pathlib
import sys


lr = pathlib.Path(sys.argv[1])
screen = lr / "motivation_search/mango_l1_screen"


def summarize(paths: list[pathlib.Path]) -> tuple[int, float, float]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        if result.get("status") != "pass":
            continue
        metric = result.get("metric_decomposition") or {}
        rows.append((float(metric.get("official_sr", 0)), float(metric.get("official_cc", 0))))
    if len(rows) != 20:
        raise RuntimeError(f"expected 20 pass cells, found {len(rows)}")
    return len(rows), 100 * sum(x for x, _ in rows) / len(rows), sum(y for _, y in rows) / len(rows)


stats = {}
for arm in ("base", "aegis"):
    stats[arm] = summarize([
        lr / f"multi_t2/{arm}/off{off}_r{advance}/result.json"
        for off in range(10) for advance in (0, 1)
    ])
for arm in ("q20", "q20_r3"):
    stats[arm] = summarize(sorted((screen / arm).glob("off*_r*/result.json")))
for arm, (n, sr, cc) in stats.items():
    print(f"{arm:7s} n={n:2d} SR={sr:5.1f} CC={cc:7.2f}")

_, base_sr, base_cc = stats["base"]
_, aegis_sr, aegis_cc = stats["aegis"]
feasible = []
for arm in ("q20", "q20_r3"):
    _, sr, cc = stats[arm]
    criteria = {
        "sr_gain_at_least_10pp": sr >= max(base_sr, aegis_sr) + 10,
        "cc_below_aegis": cc <= aegis_cc,
        "cc_at_most_half_base": cc <= 0.50 * base_cc,
    }
    for name, passed in criteria.items():
        print(f"{arm}_{name}={passed}")
    if all(criteria.values()):
        feasible.append((sr, -cc, arm))
selected = max(feasible)[2] if feasible else None
print(f"SELECTED={selected}")
print(f"CONFIRM={selected is not None}")
raise SystemExit(0 if selected else 2)
