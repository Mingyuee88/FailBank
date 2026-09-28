#!/usr/bin/env bash
#$ -N ResolvePi0
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -pe smp 4
#$ -j y
#
# Two steps, in this order on purpose:
#   1. Re-run the Pi0.5 resolution into a temp file and diff it against the committed
#      config_resolution.json. resolve_config.py was just parameterized; if that refactor
#      changed anything about the Pi0.5 path, this diff catches it before Pi0 results are
#      trusted. Identity check first, new architecture second.
#   2. Resolve Pi0 against checkpoints/pi0_vla_arena_finetuned.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export JAX_PLATFORMS=cpu          # shape resolution only; no kernels need the GPU
cd ${SE_VLA_ROOT}
PY="external/VLA-Arena/envs/openpi/.venv/bin/python"
export PYTHONPATH=src:external/VLA-Arena:.

echo "########## 1. Pi0.5 identity check ##########"
$PY roundG/pi05_smoke/resolve_config.py \
    --output /tmp/resolution_pi05_recheck.json > /tmp/pi05_recheck.stdout 2>&1 || {
      echo "PI05_RESOLVE_FAILED"; tail -20 /tmp/pi05_recheck.stdout; exit 1; }
$PY - <<'PYDIFF'
import json
a = json.load(open("roundG/pi05_smoke/config_resolution.json"))
b = json.load(open("/tmp/resolution_pi05_recheck.json"))
# architecture_family is newly added by the refactor; everything else must be identical.
added = set(b) - set(a)
assert added <= {"architecture_family"}, f"unexpected new keys: {added}"
diffs = {k: (a[k], b[k]) for k in a if a.get(k) != b.get(k)}
if diffs:
    print("PI05_IDENTITY_FAIL")
    for k, (x, y) in diffs.items():
        print(f"  {k}:\n    committed={str(x)[:200]}\n    recheck  ={str(y)[:200]}")
    raise SystemExit(1)
print(f"PI05_IDENTITY_OK  keys={len(a)}  newly_added={sorted(added)}")
print(f"  status={b['status']} family={b.get('architecture_family')} "
      f"leaves={b['checkpoint_param_leaf_count']} full_tree_match={b['full_tree_match']}")
PYDIFF

echo
echo "########## 2. Pi0 resolution ##########"
$PY roundG/pi05_smoke/resolve_config.py \
    --config pi0_vla_arena \
    --checkpoint ${SE_VLA_ROOT}/checkpoints/pi0_vla_arena_finetuned \
    --output ${SE_VLA_ROOT}/roundG/pi05_smoke/config_resolution_pi0.json \
    --source "VLA-Arena/pi0-vla-arena-fintuned@9fb1694aa8f6 (params+assets only; train_state omitted)" \
    > /tmp/pi0_resolve.stdout 2>&1 || { echo "PI0_RESOLVE_FAILED"; tail -40 /tmp/pi0_resolve.stdout; exit 1; }
$PY - <<'PYSHOW'
import json
d = json.load(open("roundG/pi05_smoke/config_resolution_pi0.json"))
print("PI0_RESOLVE status=%s family=%s model_type=%s pi05=%s horizon=%s" % (
    d["status"], d.get("architecture_family"), d["model_type"], d["pi05"], d["action_horizon"]))
print("  leaves expected=%s actual=%s matched=%s missing=%s extra=%s shape_mismatch=%s" % (
    d["expected_param_leaf_count"], d["checkpoint_param_leaf_count"],
    d["matched_param_leaf_count"], d["missing_key_count"],
    d["extra_key_count"], d["shape_mismatch_count"]))
print("  full_tree_match=%s norm_stats state=%s action=%s" % (
    d["full_tree_match"], d["norm_stats"]["state_dimension"], d["norm_stats"]["action_dimension"]))
for k, v in d["critical_layer_shapes"].items():
    print("    %-70s %s %s" % (k, v["actual"], "OK" if v["match"] else "MISMATCH"))
PYSHOW
echo "RESOLVE_JOB_DONE"
