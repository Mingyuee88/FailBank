#!/usr/bin/env bash
#$ -N InLoopCol
#$ -cwd
#$ -q gpu@HOST_A
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-47
#$ -tc 4
#$ -j y
#
# Ablation of the collection mode: what does a bank collected with the shield IN THE LOOP
# contain, and can anything be distilled from it?
#
# Derived verbatim from run_C1_collect.sh. The ONLY difference is that
# SE_VLA_SHIELD_OBSERVE_ONLY is left unset, so the shield's projection reaches the
# environment instead of being recorded as a target. Same checkpoint, same task, same
# offsets, same seed, so every cell is paired with an archived c1_collect cell.
#
# PINNED to HOST_A: every archived c1_collect and c1_collect_shieldinloop cell ran
# there (47/47 and 2/2), so pairing requires the same GPU model -- and the first attempt
# on the gpu@@GROUP group landed on a 2080Ti and died with RESOURCE_EXHAUSTED before
# the policy server could load pi0.5.
#
# WHY A 3-CELL SMOKE FIRST. The two pre-existing shield-in-loop cells
# (c1_collect_shieldinloop/off0,off1) are self-contradictory: the outcome flips hard
# (SR 0 -> 1, policy-induced CC 233/222 -> 0, 300 -> 112 chunks) while every correction
# channel reads zero -- adapter.corrections_applied=0, all_actions_forwarded_unchanged=true,
# override_delta non-zero on 0 of 112 steps, and srd's correction_applied_rate=0.0 on BOTH
# modes. No stored field distinguishes "shield computed" from "shield applied". Before
# spending 47 cells we check on 3 whether unsetting the flag actually steers; if the override
# is again all-zero while the outcome still flips, the mode is doing something other than
# steering and the ablation would measure the wrong thing.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/inloop_collect
CK=$LR/v20_fold/D_new_qw0.0/offset_0

OFF=$((SGE_TASK_ID - 1))
OUT=$OUTROOT/off$OFF
mkdir -p "$OUT" "$OUTROOT/batch"
if [ -s "$OUT/result.json" ]; then echo "SKIP off$OFF"; exit 0; fi
hostname > "$OUT/host.txt"
[ -d "$CK/params" ] || { echo "INLOOP_ABORT: no checkpoint at $CK"; exit 1; }
export SE_VLA_POLICY_CHECKPOINT_DIR="$CK"

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
# THE ONE DIFFERENCE: shield steers instead of observing.
unset SE_VLA_SHIELD_OBSERVE_ONLY || true
# BOTH flags are required: run.py gates the recorder on
#   full = _PHASE1_RECORD and _PHASE1_CANDIDATES
# and the first attempt set only PHASE1_ROOT, so 47 cells produced valid outcomes but
# zero training rows. The outcome comparison from that attempt is preserved at
# inloop_collect_norecords/ and must reproduce here (same host, deterministic).
# The recorder needs all four. Omitting RESULT_PATH leaves episode.json without a
# result_ref and build_derived_c1.py dies with KeyError; omitting EPISODE_ID leaves
# task_description empty, and since all 47 cells share one records root that makes the
# episodes unmatchable to cells after the fact (the failure run_C1_collect.sh warns about).
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="inloop_D_off${OFF}"
export SE_VLA_PHASE1_RECORD=1 SE_VLA_PHASE1_CANDIDATES=1
export SE_VLA_PHASE1_ROOT=$OUTROOT/records

echo "=== InLoop collect off=$OFF ck=$CK observe_only=UNSET on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((56000 + 20 * (SGE_TASK_ID - 1))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "INLOOP_EXIT=$rc off=$OFF"
[ -s "$OUT/result.json" ] || { echo "INLOOP_CELL_FAILED off=$OFF"; exit 1; }
# the recorder must have produced a matchable, joinable episode -- assert, do not assume
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUTROOT/records" "$OFF" <<'PYREC'
import json, sys, glob, os
root, off = sys.argv[1], sys.argv[2]
want = "inloop_D_off%s" % off
hit = None
for m in glob.glob(os.path.join(root, "episodes/*/*/episode.json")):
    try: d = json.load(open(m))
    except Exception: continue
    if d.get("task_description") == want: hit = (m, d); break
assert hit, "no episode with task_description=%s" % want
m, d = hit
rr = (d.get("result_ref") or {}).get("path")
assert rr and os.path.exists(rr), "episode %s has no usable result_ref" % want
print("INLOOP_REC off=%s episode=%s steps=%s result_ref=OK" % (off, os.path.basename(os.path.dirname(m))[:12], d.get("step_count")))
PYREC
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$OFF" <<'PYCHK'
import json, sys, os
out, off = sys.argv[1], sys.argv[2]
r = json.load(open(os.path.join(out, "result.json")))
ad = r.get("adapter") or {}; md = r.get("metric_decomposition") or {}
nz = 0; n = 0; mx = 0.0
p = os.path.join(out, "shield_steps.jsonl")
if os.path.exists(p):
    for ln in open(p, errors="ignore"):
        try: x = json.loads(ln)
        except Exception: continue
        n += 1
        od = x.get("override_delta") or []
        m = sum(abs(v) for v in od[:3]) if od else 0.0
        mx = max(mx, m); nz += (m > 1e-9)
print("INLOOP off=%s sr=%s official_cc=%s polCC=%s chunks=%s unchanged=%s corrections=%s "
      "override_steps=%d/%d max_override=%.5f" % (
      off, r.get("successes"), md.get("official_cc"), md.get("policy_induced_cc"),
      ad.get("inference_chunks"), ad.get("all_actions_forwarded_unchanged"),
      ad.get("corrections_applied"), nz, n, mx))
PYCHK
