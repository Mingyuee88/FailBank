#!/usr/bin/env python3
"""Phase0b: which qualified offsets does frozen late_margin actually rescue?

A run is RESCUED iff SR=1 (success) AND policy_induced_cost==0 AND >=1 valid CBF
trigger (srd.jsonl aegis_projection.triggered). Per protocol an offset is
rescuable only if ALL its screening seeds are rescued (no cherry-picking).
Rescuable offsets = the trainable danger-stage source for Phase 1.
"""
from __future__ import annotations
import json
from pathlib import Path

BATCH = Path("${WORK_ROOT}/lora_dagger/phase0b_rescue/batch")
OUT = BATCH / "phase0b_rescued.json"


def trigger_count(run_dir: Path) -> int | None:
    """Count real aegis overrides. srd.jsonl has row type 'aegis_projection' with a
    top-level 'triggered' flag (NOT nested under an 'aegis_projection' key), and one
    'override_candidate' row per applied override. Count aegis_projection rows whose
    triggered is truthy."""
    f = run_dir / "srd.jsonl"
    if not f.exists():
        return None
    n = 0
    for l in f.open():
        if not l.strip():
            continue
        try:
            o = json.loads(l)
        except Exception:
            continue
        if o.get("type") == "aegis_projection" and o.get("triggered"):
            n += 1
    return n


def check_run(run_dir: Path) -> dict:
    res = run_dir / "result.json"
    if not res.exists():
        return {"status": "missing"}
    try:
        r = json.loads(res.read_text())
    except Exception as e:
        return {"status": "unverifiable", "reason": f"result unreadable: {e}"}
    if r.get("status") != "pass":
        return {"status": "unverifiable", "reason": f"status={r.get('status')!r}"}
    eps = r.get("episodes") or []
    if len(eps) != 1:
        return {"status": "unverifiable", "reason": f"{len(eps)} episodes"}
    e = eps[0]
    try:
        success = bool(e["success"])
        policy = float(e["policy_induced_cost"])
        initial = float(e["initial_attributed_cost"])
        official = float(e["official_cost"])
    except (KeyError, TypeError, ValueError) as ex:
        return {"status": "unverifiable", "reason": f"episode fields: {ex}"}
    if abs(official - (initial + policy)) > 1e-6:
        return {"status": "unverifiable", "reason": "decomp provenance fail"}
    tc = trigger_count(run_dir)
    rescued = success and abs(policy) <= 1e-6 and (tc or 0) > 0
    return {"status": "rescued" if rescued else "not_rescued",
            "sr": 1 if success else 0, "policy_induced_cost": policy,
            "trigger_count": tc}


def main() -> None:
    b = json.loads((BATCH / "batch.json").read_text())
    offs, seeds = b["offsets"], b["seeds"]
    results = []
    for off in offs:
        sr = {s: check_run(BATCH / f"p0b_late_margin_off{off}_seed{s}") for s in seeds}
        present = [x for x in sr.values() if x["status"] != "missing"]
        n_res = sum(1 for x in sr.values() if x["status"] == "rescued")
        results.append({"offset": off, "n_present": len(present), "n_rescued": n_res,
                        "offset_rescuable": len(present) == len(seeds) and n_res == len(seeds),
                        "seeds": sr})
    rescuable = [r["offset"] for r in results if r["offset_rescuable"]]
    payload = {"config": b.get("config"), "n_rescuable": len(rescuable),
               "rescuable_offsets": rescuable, "detail": results}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"n_rescuable": len(rescuable), "rescuable_offsets": rescuable}, indent=2))
    for r in results:
        if r["n_present"] == 0:
            continue
        cells = []
        for s in seeds:
            x = r["seeds"][s]
            tag = {"rescued": "R", "not_rescued": ".", "unverifiable": "?", "missing": "_"}[x["status"]]
            cells.append(f"s{s}={tag}(sr={x.get('sr')},pc={x.get('policy_induced_cost')},trig={x.get('trigger_count')})")
        print(f"  off{r['offset']}: rescuable={r['offset_rescuable']}  " + " ".join(cells))


if __name__ == "__main__":
    main()
