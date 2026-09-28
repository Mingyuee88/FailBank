#!/usr/bin/env python3
"""Aggregate a complete confirmation set and save the exact figure inputs."""

from __future__ import annotations

import json
import pathlib
import sys


root = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
summary = {"protocol": {"offsets": [10, 49], "sampler_advances": [0, 1, 2], "host": "HOST_A"}, "arms": {}}
for arm in ("base", "aegis", "failbank"):
    rows = []
    for path in sorted((root / arm).glob("off*_r*/result.json")):
        result = json.loads(path.read_text())
        if result.get("status") != "pass":
            continue
        metric = result.get("metric_decomposition") or {}
        rows.append({
            "success": float(metric.get("official_sr", 0.0)),
            "cc": float(metric.get("official_cc", 0.0)),
            "path": str(path),
        })
    if len(rows) != 120:
        raise RuntimeError(f"{arm}: expected 120 pass cells, found {len(rows)}")
    summary["arms"][arm] = {
        "n": len(rows),
        "sr": 100.0 * sum(row["success"] for row in rows) / len(rows),
        "cc": sum(row["cc"] for row in rows) / len(rows),
    }
out.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
