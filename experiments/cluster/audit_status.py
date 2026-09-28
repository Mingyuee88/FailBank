#!/usr/bin/env python3
"""How many evaluated cells are actually crashed episodes being scored as failures?

A mid-episode websocket crash writes a result.json with status="fail", episodes=[] and
successes=0. Anything that reads `successes` without checking `status` counts that as a
task failure, which silently pushes an arm's SR down by however many cells crashed. The
per-cell resume rule was fixed to treat non-pass as not-done, but every reader has to
apply the same rule or the two disagree.

This enumerates every result.json under the evaluation and census trees and reports the
non-pass cells per arm and domain, so they can be deleted and re-run.
"""
import glob
import json
import os
from collections import Counter

LR = "${WORK_ROOT}/lora_dagger"

ROOTS = [
    ("v32_eval", f"{LR}/v32_eval/*/*/L*t*/off*"),
    ("v34_shieldxfer", f"{LR}/v34_shieldxfer/*/*/L*t*/off*"),
    ("v28_barrier", f"{LR}/v28_barrier/*/off*"),
    ("v26_census", f"{LR}/v26_census/*/L*t*/off*"),
    ("v8_clean", f"{LR}/v8_clean/*/off*"),
    ("v11_merged", f"{LR}/v11_merged/L*t*/off*"),
    ("v21_eval", f"{LR}/v21_eval/*/off*"),
    ("v22_axfer", f"{LR}/v22_axfer/L*t*/off*"),
    ("v23_shxfer", f"{LR}/v23_shxfer/L*t*/off*"),
    ("v25_dxfer", f"{LR}/v25_dxfer/L*t*/off*"),
    ("v9_power", f"{LR}/v9_power/*/off*"),
    ("v24_shbridge", f"{LR}/v24_shbridge/*/off*"),
]


def main():
    total = Counter()
    bad_by_group = Counter()
    bad_cells = []
    for name, pat in ROOTS:
        for cell in sorted(glob.glob(pat)):
            p = os.path.join(cell, "result.json")
            if not os.path.exists(p):
                continue
            try:
                r = json.load(open(p))
            except Exception:
                r = {"status": "unreadable"}
            # group = everything between the tree root and the offset dir
            rel = os.path.relpath(cell, LR)
            group = "/".join(rel.split("/")[:-1])
            total[group] += 1
            if r.get("status") != "pass":
                bad_by_group[group] += 1
                bad_cells.append((cell, r.get("status")))

    print(f"{'group':52s} {'cells':>6s} {'NON-PASS':>9s}")
    for g in sorted(total):
        if bad_by_group[g]:
            print(f"{g:52s} {total[g]:6d} {bad_by_group[g]:9d}   <-- re-run these")
    print()
    print(f"total cells scanned: {sum(total.values())}   non-pass: {len(bad_cells)}")
    if bad_cells:
        print()
        print("paths (delete result.json and resubmit the array; the resume rule will "
              "pick up exactly these):")
        for c, st in bad_cells:
            print(f"  {st:12s} {c}")


if __name__ == "__main__":
    main()
