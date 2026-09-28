#!/usr/bin/env python3
"""YMAK feasibility probe: do any attention heads in OUR pi0.5 localise the target object?

WHAT THIS TESTS. "Your Model Already Knows" (arXiv 2606.09749) builds a training-free CBF
filter on one empirical claim: a small number of attention heads reliably localise the
object the policy intends to approach. They report layer 12 / head 3 -- but that was
calibrated on THEIR pi0.5. Ours is pi05_vla_arena_finetuned, so the head assignment may
have drifted and must be re-derived. Copying their layer index would produce a fake
"reproduction failure".

WHY NO CODE IS NEEDED. The claim is about a property of the model, not about their
algorithm. We need our own pi0.5, the attention tensor, and ground-truth object poses --
all of which we already have: `geometry.py` already reads sim.data.body_xpos to build the
oracle ellipsoids that feed the shield. This probe reuses those same poses for a different
purpose: checking where attention peaks.

METHOD.
  1. capture   attention probs from gemma.py:271 (`probs`, layout BKGTS) via flax sow
  2. reduce    take action-query rows, keep columns that are image tokens -> patch grid
  3. project   target object world pos -> camera pixel -> patch index (ground truth)
  4. score     per (layer, head): fraction of steps whose argmax patch is within
               TOL patches of the target patch
  5. compare   against the uniform-attention baseline 1/num_patches, and against the
               DISTRACTOR object, to be sure a head tracks the target rather than
               "whatever is salient"

READING THE RESULT. Heads well above the uniform baseline -> the property holds, record
(layer, head) and proceed to a full implementation. All heads near baseline -> the
property does not survive VLA-Arena finetuning; report that as a negative result and cite
the paper in related work instead of running a broken comparison.
"""
import argparse, json, os, pathlib, sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True,
                    help="derived record set whose blobs hold the stored observations")
    ap.add_argument("--checkpoint", required=True, help="pi0.5 checkpoint (params dir's parent)")
    ap.add_argument("--limit", type=int, default=200, help="steps to score")
    ap.add_argument("--tol", type=int, default=1, help="patch-distance tolerance for a hit")
    ap.add_argument("--out", required=True, help="where to write the layer x head table")
    args = ap.parse_args()

    rec = pathlib.Path(args.records) / "derived/folds/offset_0/train.jsonl"
    if not rec.exists():
        print(f"PROBE_ABORT: no record set at {rec}")
        return 1

    # Stage 0 -- report what is available before touching the model, so a missing
    # prerequisite is named rather than surfacing as a confusing traceback later.
    have_img = have_pose = 0
    total = 0
    with rec.open() as fh:
        for line in fh:
            if total >= args.limit:
                break
            try:
                r = json.loads(line)
            except Exception:
                continue
            total += 1
            refs = r.get("raw_observation_refs") or {}
            if "agentview_image" in refs:
                have_img += 1
            # object poses are not stored in the record; they must be recomputed by
            # replaying the sim, OR read from the shield telemetry if it was logged
            if (r.get("formal_projection") or {}).get("triggered") is not None:
                have_pose += 1
    print(f"PROBE_INPUTS records={total} with_image={have_img} with_shield_row={have_pose}")
    if have_img == 0:
        print("PROBE_ABORT: no stored images in this record set")
        return 1

    print("PROBE_STAGE0_OK -- image blobs present; next stage needs the sow patch in "
          "gemma.py:271 and a sim replay for ground-truth object pixels")
    pathlib.Path(args.out).write_text(json.dumps({
        "stage": "inputs_verified",
        "records_scanned": total,
        "with_image": have_img,
        "note": "attention capture not yet wired; see module docstring for the 5 steps",
    }, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
