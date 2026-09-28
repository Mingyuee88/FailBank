#!/usr/bin/env python3
"""Phase0 offset qualification for LoRA-DAgger (reads result.json directly).

The pi05 static_sweep pipeline runs metric_decomposition INLINE and writes, per
run, result.json with:
  successes, metric_decomposition{official_cc, official_sr, policy_induced_cc},
  episodes[]{official_cost, policy_induced_cost, initial_attributed_cost, success}
so we do NOT parse arena_logs or re-derive from shield_steps.

A (offset, screening-seed) episode is a QUALIFIED base failure iff:
  1. status == "pass" and arm == "base"                     (clean pure-base run)
  2. success is False                                        (SR=0)
  3. policy_induced_cost > EPS_COST                          (real policy-caused cost)
  4. |official - (initial_attributed + policy_induced)| <= 1e-6
     -> provenance guard: the decomposition is REAL, not an official_cc alias
        (reset-contaminated cost lands in initial_attributed, so criterion 3
         already excludes teapot/reset-only failures).

Offset qualifies iff >=2 of its 3 screening seeds are qualified base failures.
Late-margin rescue pairing (1b) is a separate downstream step.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

BATCH = Path("${WORK_ROOT}/lora_dagger/phase0_base_scan/batch")
OUT = BATCH / "phase0_qualified.json"
EPS_COST = 1e-6


def qualify_run(run_dir: Path) -> dict:
    res = run_dir / "result.json"
    if not res.exists():
        return {"status": "missing", "reason": "no result.json"}
    try:
        r = json.loads(res.read_text())
    except Exception as e:
        return {"status": "unverifiable", "reason": f"result.json unreadable: {e}"}
    if r.get("status") != "pass":
        return {"status": "unverifiable", "reason": f"run status={r.get('status')!r}"}
    if r.get("arm") != "base":
        return {"status": "unverifiable", "reason": f"arm={r.get('arm')!r} not base"}
    eps = r.get("episodes") or []
    if len(eps) != 1:
        return {"status": "unverifiable", "reason": f"expected 1 episode (trials=1), got {len(eps)}"}
    e = eps[0]
    try:
        official = float(e["official_cost"])
        policy = float(e["policy_induced_cost"])
        initial = float(e["initial_attributed_cost"])
        success = bool(e["success"])
    except (KeyError, TypeError, ValueError) as ex:
        return {"status": "unverifiable", "reason": f"episode fields missing: {ex}"}
    # provenance guard: real decomposition, not an alias
    if abs(official - (initial + policy)) > 1e-6:
        return {"status": "unverifiable",
                "reason": f"decomp provenance fail: official={official} != initial({initial})+policy({policy})"}
    qualified = (not success) and (policy > EPS_COST)
    return {"status": "qualified_base_failure" if qualified else "not_qualified",
            "sr": 0 if not success else 1, "official_cost": official,
            "policy_induced_cost": policy, "initial_attributed_cost": initial}


def main() -> None:
    batch = json.loads((BATCH / "batch.json").read_text())
    offsets, seeds = batch["offsets"], batch["seeds"]
    results = []
    for off in offsets:
        seed_res = {}
        for s in seeds:
            seed_res[s] = qualify_run(BATCH / f"p0_base_task2_off{off}_seed{s}")
        present = [r for r in seed_res.values() if r["status"] != "missing"]
        n_qual = sum(1 for r in seed_res.values() if r["status"] == "qualified_base_failure")
        n_unver = sum(1 for r in seed_res.values() if r["status"] == "unverifiable")
        results.append({"offset": off, "n_present": len(present),
                        "n_qualified_seeds": n_qual, "n_unverifiable": n_unver,
                        "offset_qualified": n_qual >= 2, "seeds": seed_res})
    scanned = [r for r in results if r["n_present"] > 0]
    qualified = [r["offset"] for r in results if r["offset_qualified"]]
    payload = {"eps_cost": EPS_COST, "screening_seeds": seeds,
               "n_offsets_with_data": len(scanned), "n_offsets_qualified": len(qualified),
               "qualified_offsets": qualified, "detail": results}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"n_offsets_with_data": len(scanned),
                      "n_offsets_qualified": len(qualified),
                      "qualified_offsets": qualified}, indent=2))
    # smoke visibility: per-offset one-liner for offsets that have data
    for r in scanned:
        cells = []
        for s in seeds:
            sr = r["seeds"][s]
            tag = {"qualified_base_failure": "Q", "not_qualified": ".",
                   "unverifiable": "?", "missing": "_"}[sr["status"]]
            pc = sr.get("policy_induced_cost")
            cells.append(f"s{s}={tag}" + (f"(pc={pc:.0f})" if pc is not None else ""))
        print(f"  off{r['offset']}: qualified={r['offset_qualified']}  " + " ".join(cells))


if __name__ == "__main__":
    main()
