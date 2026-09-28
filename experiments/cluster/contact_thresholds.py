#!/usr/bin/env python3
"""Measure the contact distance PER TASK.

`cost_pair_min_distance` is a centre-to-centre distance, so the value at which contact
begins is r_a + r_b -- a property of the two objects, not a constant. RISK_THRESHOLD=0.1087
was calibrated on L1t2 (mango + white_yellow_mug, a small mug). L1t3 pairs an onion with
white_storage_box, and there the T3pre telemetry shows contact starting near 0.1855, with no
cell ever reaching 0.1087. Any fixed threshold is therefore right on one task and wrong on
the others. This recovers the empirical contact distance for whatever runs are available:
the largest distance at which a step still reports contact, versus the smallest at which it
does not.
"""
import json, glob, collections, statistics as st, pathlib

def scan(pattern, label):
    with_c, without_c = [], []
    cells = 0
    for f in glob.glob(pattern):
        seen = False
        for line in open(f, errors="ignore"):
            try: x = json.loads(line)
            except Exception: continue
            pre = x.get("pre_step_features") or {}
            if not pre and x.get("type") == "retreat_step":
                d, c = x.get("min_distance"), x.get("contacts")
            else:
                d, c = pre.get("cost_pair_min_distance"), pre.get("cost_pair_contacts")
            if d is None or c is None: continue
            seen = True
            (with_c if int(c) > 0 else without_c).append(float(d))
        if seen: cells += 1
    if not with_c and not without_c:
        print("  %-28s no per-step distance telemetry" % label); return
    line = "  %-28s cells=%d  contact-steps=%d  clear-steps=%d" % (label, cells, len(with_c), len(without_c))
    print(line)
    if with_c:
        with_c.sort()
        print("      distance WHILE touching : min %.4f  median %.4f  max %.4f"
              % (with_c[0], st.median(with_c), with_c[-1]))
    if without_c:
        without_c.sort()
        print("      distance while NOT touching: min %.4f  median %.4f"
              % (without_c[0], st.median(without_c)))
    if with_c and without_c:
        print("      => contact begins around %.4f (max touching distance)" % with_c[-1])

LR = "${WORK_ROOT}/lora_dagger"
print("=== empirical contact distance by task ===")
scan(LR + "/t3_safe3/liftN*/off*_r*/srd.jsonl", "L1t3 onion+storage_box")
scan(LR + "/c1_collect/**/srd.jsonl", "L1t2 collection bank")
scan(LR + "/xfer/L1t2/off*_r*/srd.jsonl", "L1t2 mango+yellow_mug")
scan(LR + "/xfer/L1t1/off*_r*/srd.jsonl", "L1t1")
scan(LR + "/l2_full/ours/off*_r*/srd.jsonl", "L2 mango+red_mug x2")
print()
print("RISK_THRESHOLD in build_derived_c1.py = 0.1087 (calibrated on L1t2 only)")
print("T3pre `off` arm on L1t3: min dmin seen = 0.1262, contact boundary ~0.1855")
