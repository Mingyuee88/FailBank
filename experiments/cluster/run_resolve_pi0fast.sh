#!/usr/bin/env bash
# Resolve pi0-FAST against the freshly fetched checkpoint, and gate on identity first.
#
# Order matters, same as run_resolve_pi0.sh: re-resolve Pi0.5 and diff it against the
# committed artifact BEFORE trusting anything about a new architecture. If the shared
# resolver drifted, that diff catches it here rather than inside a pi0-fast result.
#
# pi0-FAST is an AUTOREGRESSIVE head (FAST action tokenizer), unlike pi0/pi05 which are
# flow matching. That is the point of the arm: the quiet-flow guard computes a ratio of
# model.compute_loss() on quiet samples, which is head-agnostic in form but is a
# cross-entropy here rather than a flow loss. Do not call the resulting number a
# "flow ratio" when reporting it.
#
# The evaluation yaml configs/evaluation/openpi_fast.yaml names
# policy_checkpoint_dir "VLA-Arena/pi0_fast_vla_arena_low_mem_finetune", which does NOT
# exist on the Hub. The real repo is the hyphenated "pi0-fast-vla-arena-fintuned".
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export JAX_PLATFORMS=cpu          # shape resolution only
cd ${SE_VLA_ROOT}
PY="external/VLA-Arena/envs/openpi/.venv/bin/python"
export PYTHONPATH=src:external/VLA-Arena:.
CKPT=${SE_VLA_ROOT}/checkpoints/pi0_fast_vla_arena_finetuned
SHA=$(grep -o "sha=[0-9a-f]*" /tmp/fetch_fast.log 2>/dev/null | head -1 | cut -d= -f2)
SHA=${SHA:-unknown}

echo "########## 1. Pi0.5 identity re-check ##########"
$PY roundG/pi05_smoke/resolve_config.py \
    --output /tmp/resolution_pi05_recheck2.json > /tmp/pi05_recheck2.stdout 2>&1 || {
      echo "PI05_RESOLVE_FAILED"; tail -20 /tmp/pi05_recheck2.stdout; exit 1; }
$PY - <<'PYDIFF'
import json
a = json.load(open("roundG/pi05_smoke/config_resolution.json"))
b = json.load(open("/tmp/resolution_pi05_recheck2.json"))
keys = ("status", "model_type", "pi05", "action_horizon", "architecture_family",
        "checkpoint_param_leaf_count", "full_tree_match")
# A key ABSENT from the committed artifact is a field the resolver gained after that file
# was written (architecture_family is one such): additive, not drift. A key present in both
# whose value changed IS drift and must stop the run.
shared = [k for k in keys if k in a and k in b]
drift  = {k: (a[k], b[k]) for k in shared if a[k] != b[k]}
added  = {k: b.get(k) for k in keys if k not in a and k in b}
print("PI05_RECHECK", "IDENTICAL" if not drift else f"DRIFTED {drift}",
      f"(fields added since the artifact was written: {added})" if added else "")
assert not drift, "the shared resolver changed a Pi0.5 value; fix before trusting pi0-fast"
fam = added.get("architecture_family", "pi05")
assert fam == "pi05", f"newly resolved architecture_family for the Pi0.5 checkpoint is {fam!r}, expected pi05"
PYDIFF

echo "########## 2. pi0-FAST resolution ##########"
[ -d "$CKPT/params" ] || { echo "PI0FAST_ABORT: no params at $CKPT"; exit 1; }
$PY roundG/pi05_smoke/resolve_config.py \
    --config pi0_fast_vla_arena \
    --checkpoint "$CKPT" \
    --output ${SE_VLA_ROOT}/roundG/pi05_smoke/config_resolution_pi0fast.json \
    --source "VLA-Arena/pi0-fast-vla-arena-fintuned@${SHA} (params+assets only; train_state omitted)" \
    > /tmp/pi0fast_resolve.stdout 2>&1 || { echo "PI0FAST_RESOLVE_FAILED"; tail -40 /tmp/pi0fast_resolve.stdout; exit 1; }
$PY - <<'PYSHOW'
import json
d = json.load(open("roundG/pi05_smoke/config_resolution_pi0fast.json"))
print("PI0FAST_RESOLVE status=%s family=%s model_type=%s pi05=%s horizon=%s" % (
    d["status"], d.get("architecture_family"), d["model_type"], d["pi05"], d["action_horizon"]))
print("  leaves expected=%s actual=%s matched=%s missing=%s extra=%s shape_mismatch=%s" % (
    d["expected_param_leaf_count"], d["checkpoint_param_leaf_count"],
    d["matched_param_leaf_count"], d["missing_key_count"],
    d["extra_key_count"], d["shape_mismatch_count"]))
print("  full_tree_match=%s norm_stats state=%s action=%s" % (
    d["full_tree_match"], d["norm_stats"]["state_dimension"], d["norm_stats"]["action_dimension"]))
assert d["status"] == "pass", "pi0-fast resolution did not pass"
assert d.get("architecture_family") == "pi0_fast", \
    "family resolved to %r, not pi0_fast -- wrong checkpoint or config" % d.get("architecture_family")
PYSHOW
echo "PI0FAST_RESOLVE_DONE"
