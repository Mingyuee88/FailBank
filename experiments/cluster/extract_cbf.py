#!/usr/bin/env python3
"""Extract (observation, nominal action, CBF correction) triples across four tasks.

Predicting a scalar distance throws away direction: it says "danger" but not "which way
out", which is why a fixed +z lift helped on L1-T1 and hurt on L1-T4 -- the geometry
happened to line up in one case. The CBF's own output is a vector and is by construction
the minimal edit that keeps the task moving, so cloning it keeps both the direction and
the progress that the hand-built retreat discards.

This pass only extracts and characterises. If the corrections turn out to be dominated by
oracle geometry that is invisible in the images, the clone will fail to transfer exactly
the way the distance head did, and that is worth knowing before training anything.
"""
import json, pathlib, collections
import numpy as np

BANKS = [("L1t2", "r2_collect"), ("L1t2", "r3_collect"),
         ("L1t1", "mcol_t1"), ("L1t3", "mcol_t3"), ("L1t4", "mcol_t4")]
LR = pathlib.Path("${WORK_ROOT}/lora_dagger")
OUT = LR / "cbf_clone"
KEYS = ["observation/image", "observation/wrist_image", "observation/state"]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    stats = collections.Counter()
    for task, bank in BANKS:
        root = LR / bank / "records"
        if not root.is_dir(): stats["missing:" + bank] += 1; continue
        for meta in sorted(root.glob("episodes/*/*/raw_steps.jsonl")):
            for line in meta.open():
                try: r = json.loads(line)
                except Exception: continue
                mi = r.get("model_input_refs") or {}
                fp = r.get("formal_projection") or {}
                nom = r.get("nominal_action")
                tr = fp.get("translation")
                if not all(k in mi for k in KEYS) or nom is None or tr is None:
                    stats["incomplete"] += 1; continue
                nom3 = np.asarray(nom[:3], dtype=np.float64)
                cbf3 = np.asarray(tr[:3], dtype=np.float64)
                delta = cbf3 - nom3
                rows.append({"task": task, "episode": meta.parent.name,
                             "image": mi[KEYS[0]]["path"], "wrist": mi[KEYS[1]]["path"],
                             "state": mi[KEYS[2]]["path"], "root": str(root),
                             "nominal": nom3.tolist(), "delta": delta.tolist(),
                             "active": int(fp.get("active_constraints") or 0),
                             "min_barrier": float(fp.get("min_barrier") or 0.0),
                             "dist": float((r.get("runtime_info") or {}).get("cost_pair_min_distance") or -1)})
                stats["kept"] += 1
    if not rows:
        print("no rows", stats); return 1
    d = np.array([r["delta"] for r in rows])
    nrm = np.linalg.norm(d, axis=1)
    act = np.array([r["active"] for r in rows])
    task = np.array([r["task"] for r in rows])
    print("extracted %d steps  %s" % (len(rows), dict(stats)))
    print()
    print("correction magnitude across all steps:")
    print("    zero (|delta|<1e-9): %d (%.0f%%)" % ((nrm < 1e-9).sum(), 100.0*(nrm < 1e-9).mean()))
    nz = nrm[nrm >= 1e-9]
    if len(nz):
        for q in (50, 75, 90, 99):
            print("    nonzero p%-2d  %.4f" % (q, np.percentile(nz, q)))
    print()
    print("per task -- how much signal is there to clone?")
    for t in sorted(set(task.tolist())):
        s = task == t
        nzt = nrm[s] >= 1e-9
        print("  %-6s steps=%5d  CBF active %5.1f%%  nonzero correction %5.1f%%  mean|delta| %.4f"
              % (t, s.sum(), 100.0*(act[s] > 0).mean(), 100.0*nzt.mean(), nrm[s][nzt].mean() if nzt.any() else 0))
    print()
    # direction structure: is the correction mostly one axis, or genuinely 3-D?
    dn = d[nrm >= 1e-9]
    unit = dn / np.linalg.norm(dn, axis=1, keepdims=True)
    print("correction direction, mean |component| of the unit vector:")
    print("    x %.3f   y %.3f   z %.3f" % tuple(np.abs(unit).mean(axis=0)))
    print("    -> a fixed +z lift can only ever match the z component;")
    print("       whatever mass sits on x and y is direction the hand-built retreat cannot express")
    with (OUT/"index.jsonl").open("w") as fh:
        for r in rows: fh.write(json.dumps(r) + "\n")
    print()
    print("wrote %s/index.jsonl" % OUT)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
