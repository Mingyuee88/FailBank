#!/usr/bin/env python3
"""Extract (observation, oracle-distance) pairs across FOUR tasks.

The proprioceptive head reached MAE 0.0121 with episodes held out, then scored SR 0 on
L1-T1 and L1-T4 at deployment. The reason was never model capacity: every bank recorded
the same task, so "held-out episode" still meant "same layout". 8-D encoder readings carry
no information about where a hazard is; within one layout they can stand in for it, across
layouts they cannot.

With four tasks collected, the split that matters becomes possible: hold out a whole TASK.
Anything that does not survive that will not survive deployment either.
"""
import json, pathlib, collections
import numpy as np

BANKS = [
    ("L1t2", "${WORK_ROOT}/lora_dagger/r2_collect/records"),
    ("L1t2", "${WORK_ROOT}/lora_dagger/r3_collect/records"),
    ("L1t1", "${WORK_ROOT}/lora_dagger/mcol_t1/records"),
    ("L1t3", "${WORK_ROOT}/lora_dagger/mcol_t3/records"),
    ("L1t4", "${WORK_ROOT}/lora_dagger/mcol_t4/records"),
]
OUT = pathlib.Path("${WORK_ROOT}/lora_dagger/dist_head_multi")
KEYS = ["observation/image", "observation/wrist_image", "observation/state"]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    per_task = collections.Counter()
    skipped = collections.Counter()
    for task, root in BANKS:
        root = pathlib.Path(root)
        if not root.is_dir():
            skipped["missing_bank:" + task] += 1; continue
        for meta in sorted(root.glob("episodes/*/*/raw_steps.jsonl")):
            for line in meta.open():
                try: r = json.loads(line)
                except Exception: continue
                mi = r.get("model_input_refs") or {}
                ri = r.get("runtime_info") or {}
                d = ri.get("cost_pair_min_distance")
                if d is None or not all(k in mi for k in KEYS):
                    skipped["incomplete"] += 1; continue
                rows.append({"task": task, "episode": meta.parent.name,
                             "step": int(r.get("step_index", -1)),
                             "image": mi[KEYS[0]]["path"], "wrist": mi[KEYS[1]]["path"],
                             "state": mi[KEYS[2]]["path"], "dist": float(d),
                             "contacts": int(ri.get("cost_pair_contacts") or 0),
                             "root": str(root)})
                per_task[task] += 1
    if not rows:
        print("no rows"); return 1
    print("extracted %d steps   skipped %s" % (len(rows), dict(skipped)))
    print()
    print("per task:")
    for t in sorted(per_task):
        sub = [r for r in rows if r["task"] == t]
        d = np.array([r["dist"] for r in sub]); c = np.array([r["contacts"] for r in sub])
        touch = d[c > 0]
        print("  %-6s steps=%5d  episodes=%3d  dist p5=%.4f p50=%.4f p95=%.4f  contact steps=%d (%.0f%%)"
              % (t, len(sub), len({r["episode"] for r in sub}),
                 np.percentile(d, 5), np.percentile(d, 50), np.percentile(d, 95),
                 len(touch), 100.0*len(touch)/len(sub)))
        if len(touch):
            print("         contact distance: max %.4f   -- the trigger band for this task" % touch.max())
    print()
    print("the per-task contact bands differ, which is precisely why one hand-set threshold")
    print("(0.21, tuned on L1t3) collapsed d021 on L1-T1 and L1-T2.")
    with (OUT/"index.jsonl").open("w") as fh:
        for r in rows: fh.write(json.dumps(r) + "\n")
    print()
    print("wrote %s/index.jsonl (%d rows, %d tasks)" % (OUT, len(rows), len(per_task)))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
