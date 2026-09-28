#!/usr/bin/env python
"""Offline action-support annotator.

For every learning record, re-ask the policy the exact question it was asked at
collection time, draw K fresh action chunks, and locate the teacher's corrected action
relative to that sample cloud. Produces the per-record learnability score that the
curriculum stages on.

Why offline rather than in the rollout: `SE_VLA_VLSA_DISPERSION` cannot run with
`SE_VLA_PHASE1_RECORD=1` (run.py:93 -- the phase-1 InferCapture pops by call index and
the extra sampling calls desynchronise it), and the probe is not passive anyway (extra
inference advances the sampler, so a probed episode diverges from an un-probed one).
Annotating offline touches no rollout, is replayable, and -- decisively for a curriculum --
can be re-run against a NEW policy each round to re-assign stages as the support moves.

The formula is copied from `_vlsa_dispersion` in roundG/pi05_stage2/run.py so the numbers
are comparable with E10/E11: first action of the chunk, translation channels only.
"""
import argparse, json, os, pathlib, sys, time
import numpy as np


def load_blob(root: pathlib.Path, ref: dict):
    """Load one array from the content-addressed blob store."""
    if ref is None:
        return None
    p = ref.get("path")
    if p is None:
        return None
    for cand in (root / p, root / (p + ".npy")):
        if cand.exists():
            return np.load(cand, allow_pickle=False)
    raise FileNotFoundError(f"blob missing: {root / p}")


def build_prompt_index(records_root: pathlib.Path):
    """episode_id -> task_description, from the raw step logs (derived records drop it)."""
    idx = {}
    epdir = records_root / "episodes"
    for raw in epdir.glob("*/*/raw_steps.jsonl"):
        try:
            with raw.open() as fh:
                for line in fh:
                    r = json.loads(line)
                    ep, td = r.get("episode_id"), r.get("task_description")
                    if ep and td:
                        idx[ep] = td
                        break
        except Exception:
            continue
    return idx


def dispersion(client, element, nominal, executed, k):
    firsts = []
    for _ in range(k):
        ch = client.infer(element)["actions"]
        firsts.append(np.asarray(ch[0], dtype=float)[:3])
    F = np.stack(firsts)
    nom = np.asarray(nominal, dtype=float)[:3]
    exe = np.asarray(executed, dtype=float)[:3]
    pair = [float(np.linalg.norm(F[i] - F[j]))
            for i in range(len(F)) for j in range(i + 1, len(F))]
    d_exe = [float(np.linalg.norm(exe - f)) for f in F]
    d_nom = [float(np.linalg.norm(nom - f)) for f in F]
    return {
        "k": int(k),
        "spread_mean": float(np.mean(pair)) if pair else 0.0,
        "spread_max": float(max(pair)) if pair else 0.0,
        "d_executed_min": float(min(d_exe)),
        "d_executed_mean": float(np.mean(d_exe)),
        "d_nominal_min": float(min(d_nom)),
        "d_nominal_mean": float(np.mean(d_nom)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-root", required=True, help="…/phase1_collection/records")
    ap.add_argument("--fold", required=True, help="e.g. offset_0")
    ap.add_argument("--split", default="train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="verify blob/prompt reconstruction without contacting a policy")
    ap.add_argument("--nominal-gate", type=float, default=0.25,
                    help="median d_nominal_min above this fails the run: the reconstructed "
                         "observation is not the one the policy actually saw")
    a = ap.parse_args()

    root = pathlib.Path(a.records_root)
    blobs = root / "blobs"
    src = root / "derived" / "folds" / a.fold / a.split
    if not src.exists():
        sys.exit(f"no such split: {src}")

    prompts = build_prompt_index(root)
    print(f"prompt index: {len(prompts)} episodes", flush=True)

    client = None
    if not a.dry_run:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        client = WebsocketClientPolicy(host=a.host, port=a.port)
        print(f"connected to policy server {a.host}:{a.port}", flush=True)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = ok = miss_prompt = miss_blob = 0
    dnom = []
    t0 = time.time()
    with out.open("w") as fo:
        with src.open() as fh:
            for line in fh:
                if a.limit and n >= a.limit:
                    break
                r = json.loads(line)
                n += 1
                refs = r.get("observation_refs") or {}
                ep = r.get("episode_id")
                prompt = prompts.get(ep)
                if prompt is None:
                    miss_prompt += 1
                    continue
                try:
                    element = {k: load_blob(blobs, v) for k, v in refs.items()}
                except FileNotFoundError:
                    miss_blob += 1
                    continue
                if any(v is None for v in element.values()) or not element:
                    miss_blob += 1
                    continue
                element["prompt"] = prompt
                rec = {"record_id": r.get("record_id"), "episode_id": ep,
                       "offset": r.get("offset"), "step_index": r.get("step_index"),
                       "teacher_bucket": (r.get("teacher") or {}).get("bucket"),
                       "min_barrier": (r.get("formal_projection") or {}).get("min_barrier"),
                       "eventual_success": r.get("eventual_success")}
                if a.dry_run:
                    rec["shapes"] = {k: list(np.asarray(v).shape)
                                     for k, v in element.items() if k != "prompt"}
                else:
                    d = dispersion(client, element, r["nominal_action"], r["executed_action"], a.k)
                    rec.update(d)
                    dnom.append(d["d_nominal_min"])
                fo.write(json.dumps(rec) + "\n")
                ok += 1
                if ok % 200 == 0:
                    print(f"  {ok} annotated ({time.time()-t0:.0f}s)", flush=True)

    print(f"records={n} annotated={ok} missing_prompt={miss_prompt} missing_blob={miss_blob} "
          f"elapsed={time.time()-t0:.0f}s", flush=True)
    if miss_prompt or miss_blob:
        sys.exit(f"FAIL: {miss_prompt} records had no prompt, {miss_blob} had no blob -- "
                 "reconstruction is incomplete and the annotation would be biased")
    if not a.dry_run:
        med = float(np.median(dnom))
        print(f"SELF-CHECK median d_nominal_min = {med:.4f} "
              f"(E10 in-rollout reference: mean 0.0733)", flush=True)
        # The policy's own proposal must land inside its own sample cloud. If it does not,
        # the reconstructed observation is not the one the policy was actually asked about
        # and every support number here is noise. This is the only check that can catch it.
        if med > a.nominal_gate:
            sys.exit(f"FAIL: median d_nominal_min {med:.4f} > {a.nominal_gate}; the "
                     "reconstruction does not reproduce the policy's own action")
        print("SELF-CHECK PASSED", flush=True)


if __name__ == "__main__":
    main()
