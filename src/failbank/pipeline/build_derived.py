#!/usr/bin/env python3
"""Stage 2a: raw Stage-1 episodes -> derived, teacher-weighted, risk-staged records.

    failbank-build-derived --records-root <collection>/records [--folds 0]

For every step of every COMPLETE episode this computes, from the recorded rollout only
(no critic, no counterfactual):

* ``outcomes`` over horizons H1/H2/H4 and an ``aftermath`` window (recovery, stall,
  repeated triggers);
* ``teacher`` = ``teacher_quality()``: the admission gate and a bucketed weight
  (reject 0 / low 0.25 / med 0.60 / high 1.00) for triggered steps;
* ``quiet = not triggered``;
* ``curriculum_stage`` from ``steps_to_first_risk`` = (row index of the episode's first
  step with ``cost_pair_min_distance < RISK_THRESHOLD``) - ``step_index``:
  S1_early (>= 30), S2_mid (15..29), S3_emergency (0..14), POST_crossing (< 0),
  NO_RISK (the episode never crosses).

  Caveat, kept for reproducibility: the row index counts from 0 while ``step_index`` starts
  at 10 (the evaluator's settle steps bypass the step seam), so the label is the true lead
  minus 10. In true steps before the crossing: S1 >= 40, S2 25..39, S3 10..24, and
  POST_crossing includes the 10 steps immediately before the crossing.

Episodes are NOT filtered on outcome: failures are what this collection is for.

Records are split episode-wise by initial-state offset. For each held-out offset ``o``
the fold ``derived/folds/offset_<o>/`` holds ``train.jsonl`` (all other offsets) and
``validation.jsonl`` (offset ``o``); the validation split is the held-out set the policy
update's guard is evaluated on.

Provenance of the constants. ``Thresholds`` and the bucket edges in ``teacher_quality``
are design-time constants, not fitted to data. ``RISK_THRESHOLD = 0.1087`` was chosen as a
collision warning line on one task (L1-T2) and reused for staging; it was not re-calibrated
per task. They are kept exactly as used for the paper so that records stay commensurable.

Historical note: this module merges the original success-only ``build_derived.py`` and its
C1 wrapper; the teacher/outcome/quiet functions below are unchanged from the original.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

SCHEMA = "se_hwm_derived_record/v1"
HORIZONS = (1, 2, 4)
GOOD_SIGN = -1  # signed_distance q smaller = closer to goal = progress


@dataclass(frozen=True)
class Thresholds:
    margin_safe: float = 0.05          # forward barrier >= this = fully safe
    min_barrier_floor: float = 0.0     # violation if below
    min_risk_improvement: float = 0.0  # gate: forward min_barrier_H1 >= this (or recovery)
    progress_stall_eps: float = 2e-3
    progress_good: float = 1e-2
    recovery_barrier: float = 0.0
    recovery_progress: float = 2e-3
    recovery_clean_steps: int = 2
    pair_margin_gap: float = 1e-3
    aftermath_k: int = 8


def read_json(p: Path) -> dict: 
    with p.open() as f: return json.load(f)
def read_jsonl(p: Path) -> list[dict]:
    with p.open() as f: return [json.loads(l) for l in f if l.strip()]
def write_jsonl(p: Path, rows: Iterable[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows: f.write(json.dumps(r, separators=(",",":"), sort_keys=True)+"\n")
def stable_id(*parts) -> str:
    return hashlib.sha256("|".join(map(str,parts)).encode()).hexdigest()[:24]
def finite(v, default=math.nan) -> float:
    try:
        v=float(v); return v if math.isfinite(v) else default
    except (TypeError,ValueError): return default

def proj(s): return s.get("formal_projection") or {}
def barrier(s): return finite(proj(s).get("min_barrier"))
def triggered(s): return bool(proj(s).get("triggered", False))
def proj_ok(s): return str(proj(s).get("status","")).lower() in {"projected","nominal_safe","feasible","ok","solved","optimal","success"}
def step_cost(s): return max(0.0, finite((s.get("runtime_info") or {}).get("cost"), default=0.0))
def signed_progress(s): return GOOD_SIGN * finite(s.get("progress_delta"), default=0.0)

def find_candidate(s, name):
    for c in s.get("projection_candidates") or []:
        if c.get("name")==name: return c
    return None
def horizon_outcome(steps, t, h):
    win=steps[t:min(t+h,len(steps))]
    bars=[barrier(s) for s in win]; vb=[x for x in bars if math.isfinite(x)]
    return {"horizon":h,"available_steps":len(win),
            "any_cost":any(step_cost(s)>0 for s in win),
            "min_barrier":(min(vb) if vb else None),
            "margin_delta":(vb[-1]-vb[0] if len(vb)>=2 else 0.0),
            "progress_delta":sum(signed_progress(s) for s in win),
            "trigger_count":sum(triggered(s) for s in win)}

def has_recovery(steps, t, th):
    aft=steps[t+1:min(t+1+th.aftermath_k,len(steps))]; run=0; cum=0.0
    for s in aft:
        cum+=signed_progress(s); b=barrier(s)
        clean=(not triggered(s) and math.isfinite(b) and b>=th.recovery_barrier)
        run=run+1 if clean else 0
        if run>=th.recovery_clean_steps and cum>=th.recovery_progress: return True
    return False

def aftermath(steps, t, th):
    aft=steps[t+1:min(t+1+th.aftermath_k,len(steps))]
    prog=sum(signed_progress(s) for s in aft)
    return {"k":th.aftermath_k,"available_steps":len(aft),
            "repeated_trigger_rate":(sum(triggered(s) for s in aft)/len(aft) if aft else 0.0),
            "progress_delta":prog,"stalled":prog<=th.progress_stall_eps,
            "recovery":has_recovery(steps,t,th)}

def teacher_quality(step, outcomes, aft, th):
    """OBSERVABLE-consequence teacher weight (no counterfactual b_nom, no critic)."""
    b_exec=barrier(step)
    mb_h1=outcomes["H1"]["min_barrier"]; mb_h4=outcomes["H4"]["min_barrier"]
    executed=np.asarray(step["executed_action"],dtype=np.float32)
    delta=finite(find_candidate(step,"late_margin").get("action_delta_norm") if find_candidate(step,"late_margin") else None, default=0.0)
    # forward safety: how safe the executed action kept the next H1 window (observed)
    fwd=mb_h1 if mb_h1 is not None else b_exec
    forward_safety=float(np.clip((fwd)/th.margin_safe,0.0,1.0)) if math.isfinite(finite(fwd)) else 0.0
    prog=float(outcomes["H4"]["progress_delta"])
    if aft["stalled"]: progress_preservation=0.0
    elif prog>=th.progress_good: progress_preservation=1.0
    else: progress_preservation=0.5
    h4=outcomes["H4"]
    gate=(triggered(step) and proj_ok(step) and bool(np.isfinite(executed).all())
          and delta>1e-6
          and ((mb_h1 is not None and mb_h1>=th.min_risk_improvement) or aft["recovery"]))
    score=(0.40*forward_safety + 0.30*progress_preservation + 0.20*float(aft["recovery"])
           + 0.10*(1.0-aft["repeated_trigger_rate"])
           - 0.30*float(h4["any_cost"])
           - 0.20*float(h4["min_barrier"] is not None and h4["min_barrier"]<th.min_barrier_floor))
    score=float(np.clip(score,0.0,1.0))
    if not gate or score<0.25: bucket,weight="reject",0.0
    elif score<0.45: bucket,weight="low",0.25
    elif score<0.70: bucket,weight="med",0.60
    else: bucket,weight="high",1.00
    return {"formula_version":"observable_teacher_v2","gate":gate,"score":score,"bucket":bucket,
            "weight":weight,"forward_safety":forward_safety,"forward_min_barrier_h1":mb_h1,
            "progress_preservation":progress_preservation,"recovery":bool(aft["recovery"]),
            "uses_critic_prediction":False}

def resolve_offset_success(episode_dir):
    ej=read_json(episode_dir/"episode.json")
    rr=(ej.get("result_ref") or {}); rp=rr.get("path")
    if not rp or not Path(rp).exists(): raise KeyError(f"no result_ref for {episode_dir.name}")
    r=read_json(Path(rp)); e=(r.get("episodes") or [{}])[0]
    off=r.get("init_state_offset")
    if off is None: raise KeyError(f"no init_state_offset in {rp}")
    return str(off), bool(e.get("success")), float(e.get("policy_induced_cost",0.0) or 0.0), ej

def build_episode(episode_dir, offset, success, th):
    steps=read_jsonl(episode_dir/"raw_steps.jsonl")
    eid=episode_dir.name
    records=[]
    for t,step in enumerate(steps):
        outcomes={f"H{h}":horizon_outcome(steps,t,h) for h in HORIZONS}
        aft=aftermath(steps,t,th)
        q=teacher_quality(step,outcomes,aft,th)
        rec={"schema_version":SCHEMA,"record_id":stable_id(eid,t),"episode_id":eid,"offset":offset,
             "step_index":int(step["step_index"]),"infer_call_index":step.get("infer_call_index"),
             "triggered":triggered(step),"observation_refs":step["model_input_refs"],
             "raw_observation_refs":step.get("raw_observation_refs"),
             "nominal_action":step["nominal_action"],"executed_action":step["executed_action"],
             "nominal_action_chunk_ref":step["nominal_action_chunk_ref"],
             "formal_projection":step["formal_projection"],"outcomes":outcomes,"aftermath":aft,
             "eventual_success":success,"teacher":q,"quiet":not triggered(step)}
        records.append(rec)
    return records



# --------------------------------------------------------------------------- risk staging

RISK_THRESHOLD = 0.1087
STAGES = (("S1_early", 30, 10**9), ("S2_mid", 15, 30),
          ("S3_emergency", 0, 15), ("POST_crossing", -10**9, 0))


def first_risk_step(episode_dir: Path):
    """Index of the first step whose cost_pair distance is inside the risk threshold."""
    f = episode_dir / "raw_steps.jsonl"
    if not f.exists():
        return None
    for i, line in enumerate(f.open()):
        try:
            d = (json.loads(line).get("runtime_info") or {}).get("cost_pair_min_distance")
        except Exception:
            continue
        if d is not None and d < RISK_THRESHOLD:
            return i
    return None


def stage_for(lead):
    if lead is None:
        return "NO_RISK"
    for name, lo, hi in STAGES:
        if lo <= lead < hi:
            return name
    return "NO_RISK"


def build_all(root: Path, th: Thresholds):
    all_rec = []
    kept = dropped = 0
    for ej in sorted(root.glob("episodes/*/*/episode.json")):
        ed = ej.parent
        if not (ed / "COMPLETE").exists():
            dropped += 1
            continue
        offset, success, _pc, _ = resolve_offset_success(ed)
        recs = build_episode(ed, offset, success, th)
        fr = first_risk_step(ed)
        for r in recs:
            lead = None if fr is None else fr - int(r["step_index"])
            r["steps_to_first_risk"] = lead
            r["curriculum_stage"] = stage_for(lead)
            r["risk_threshold"] = RISK_THRESHOLD
        all_rec.extend(recs)
        kept += 1
    return all_rec, kept, dropped


def write_folds(root: Path, all_rec, folds=None):
    offsets = sorted({r["offset"] for r in all_rec}, key=lambda x: int(x))
    held_out = offsets if not folds else [str(f) for f in folds]
    missing = sorted(set(held_out) - set(offsets), key=int)
    if missing:
        raise ValueError(f"requested folds {missing} have no episodes (offsets present: {offsets})")
    derived = root / "derived"
    for ho in held_out:
        fr_dir = derived / "folds" / f"offset_{ho}"
        train = [r for r in all_rec if r["offset"] != ho]
        val = [r for r in all_rec if r["offset"] == ho]
        write_jsonl(fr_dir / "train.jsonl", train)
        write_jsonl(fr_dir / "validation.jsonl", val)
        fr_dir.mkdir(parents=True, exist_ok=True)
        (fr_dir / "manifest.json").write_text(json.dumps({
            "schema_version": SCHEMA, "heldout_offset": ho,
            "outcome_filter": "DISABLED -- failures are the point of this collection",
            "risk_threshold": RISK_THRESHOLD,
            "stage_counts_train": dict(Counter(r["curriculum_stage"] for r in train)),
            "train_records": len(train), "validation_records": len(val),
            "train_triggered": sum(r["triggered"] for r in train),
        }, indent=2, sort_keys=True))
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "schema_version.txt").write_text(SCHEMA + "\n")
    return held_out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records-root", type=Path, required=True,
                    help="collection root holding episodes/ and blobs/")
    ap.add_argument("--folds", type=int, nargs="*", default=None,
                    help="held-out offsets to write (default: one fold per offset)")
    a = ap.parse_args(argv)
    root = a.records_root.resolve()
    th = Thresholds()

    all_rec, kept, dropped = build_all(root, th)
    if kept == 0:
        raise SystemExit(f"no COMPLETE episodes under {root}/episodes")
    trig = Counter(r["curriculum_stage"] for r in all_rec if r["triggered"])
    print(f"episodes kept: {kept}  (incomplete dropped: {dropped})")
    print(f"total records: {len(all_rec)} | triggered: {sum(r['triggered'] for r in all_rec)}")
    print(f"eventual_success distribution: {dict(Counter(r['eventual_success'] for r in all_rec))}")
    print(f"stages (all records): {dict(Counter(r['curriculum_stage'] for r in all_rec))}")
    print(f"stages (triggered only, i.e. carrying a teacher target): {dict(trig)}")
    print(f"triggered teacher buckets: "
          f"{dict(Counter(r['teacher']['bucket'] for r in all_rec if r['triggered']))}")
    if not any(not r["eventual_success"] for r in all_rec):
        print("WARNING: no failure episodes in this collection")
    if trig.get("S1_early", 0) == 0:
        print("WARNING: no triggered S1_early records -- this collection carries no trainable correction")

    held_out = write_folds(root, all_rec, a.folds)
    print(f"wrote {len(held_out)} held-out folds under {root}/derived/folds/")
    print("DERIVED_BUILT")


if __name__ == "__main__":
    main()
