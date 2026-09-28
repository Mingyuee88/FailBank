#!/usr/bin/env python3
"""E14 AEGIS validity gate on smoke cells. Prints GATE_PASS / GATE_FAIL with per-cell reasons."""
import json, os, sys
cells = sys.argv[1:]; ok = True
for d in cells:
    r = json.load(open(d + "/result.json")) if os.path.exists(d + "/result.json") else None
    a = None
    if os.path.exists(d + "/srd.jsonl"):
        for line in open(d + "/srd.jsonl", errors="ignore"):
            if '"vlsa_audit"' in line:
                a = json.loads(line); break
    reasons = []
    if not r or r.get("status") != "pass": reasons.append("status")
    if not a: reasons.append("no_audit")
    else:
        if a.get("perception_ok") is not True: reasons.append("perception")
        if not (a.get("points_filtered") or 0) > 0: reasons.append("no_points")
        name = str(a.get("obstacle_name") or "").lower()
        if not any(k in name for k in ("stove", "candle", "burner", "flame", "cooktop")): reasons.append("wrong_obstacle:" + name)
        if not (a.get("constraint_active_steps") or 0) > 0 or not (a.get("override_norm_sum") or 0) > 0: reasons.append("shield_inert")
    print(d, "OK" if not reasons else "FAIL " + ",".join(reasons),
          {k: (a or {}).get(k) for k in ("obstacle_name", "points_filtered", "constraint_active_steps", "override_norm_sum")})
    ok &= not reasons
print("GATE_PASS" if ok else "GATE_FAIL")
