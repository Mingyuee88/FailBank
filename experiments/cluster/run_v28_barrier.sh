#!/usr/bin/env bash
#$ -N v28_barr
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 2
#$ -t 1-141
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v28_barrier/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v28_barrier/batch/j.$JOB_ID.$TASK_ID.out
#
# v28_barrier -- the causal test of the misalignment mechanism.
#
# WHAT THE EVIDENCE ACTUALLY SAYS. A first pass found that the shield destroys episodes
# in which the object was far from the gripper at the moments the shield fired
# (AUC 0.889, p=0.0006). That measurement was taken on shield-generated trajectories at
# shield-selected times, so the separation could equally well be a CONSEQUENCE of the
# shield's own interference. Recomputing on the steps STRICTLY BEFORE the first
# projection -- causally untouched, because this codebase is bit-deterministic on a
# pinned host and the shield changes nothing until it first fires -- the association
# collapses and reverses (AUC 0.309, p=0.089). That reading is withdrawn.
#
# What survives pre-treatment is different and sharper:
#
#   pre_d_obj  (object-to-hazard distance BEFORE any intervention)
#              destroyed median 0.157 m   kept median 0.175 m   AUC 0.177   p=0.0041
#   the first projection fires while the object is still ungrasped in 36 of 36 episodes,
#   at step 11-19 of ~120-300
#
# So the shield always intervenes during the REACH, never during transport, and the
# episodes it destroys are the ones whose target happens to sit closer to the hazard.
# Meanwhile the benchmark's cost is a conjunction -- hazard falls AND object contacts
# hazard AND gripper contacts hazard -- that a reach cannot satisfy. A gripper-centred
# barrier is therefore blocking approaches that were never going to incur cost.
#
# THE TEST. Same shield, same alpha/radii/margin, same host, same 47 offsets, only the
# barrier CENTRE changes:
#
#   eef     unchanged. Present as a REGRESSION GATE: it must reproduce the historical
#           numbers cell for cell, or the patch changed the default path and every
#           comparison in this job is void.
#   object  barrier on the manipulated object, active only while the object is within
#           SE_VLA_AEGIS_GRASP_GATE_M of the gripper. The gate is physics, not tuning:
#           the projection edits the GRIPPER's velocity and only moves the object while
#           the object is held.
#   dual    both barriers in sequence -- keep the gripper from knocking the hazard AND
#           keep the carried object away from it.
#
# PREDICTION, fixed before the run. If misalignment is the mechanism, `object` stops
# destroying successes during the reach (destroyed count 9/36 falls sharply) while
# keeping the repairs that happen during transport. If `object` destroys just as many,
# or loses all the repairs, the mechanism story is wrong and I will say so.
#
# READOUT: repair rate and retention rate reported separately, never netted; paired
# McNemar against the eef arm on the same offsets; obj_travel checked in BOTH directions
# (a barrier that simply freezes the arm buys CC by not doing the task).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

OFFS=(0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47)
NOFF=${#OFFS[@]}   # 47
ARMS=(eef object dual)
i=$((SGE_TASK_ID - 1))
ARM=${ARMS[$((i / NOFF))]}
OFF=${OFFS[$((i % NOFF))]}

OUT=$LR/v28_barrier/$ARM/off$OFF
mkdir -p "$OUT" $LR/v28_barrier/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
# base policy under the shield -- exactly the configuration that produced the 9/36
# destruction rate, with the single variable of interest changed
unset SE_VLA_POLICY_CHECKPOINT_DIR || true
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_AEGIS_BARRIER_CENTER="$ARM"
export SE_VLA_AEGIS_GRASP_GATE_M=0.12
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="v28_${ARM}_off${OFF}"

echo "=== v28 barrier=$ARM off=$OFF on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((48000 + 20 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "V28_EXIT=$rc arm=$ARM off=$OFF"

# ---- audit: which barrier actually fired, counted by the code that fired it ----
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$ARM" "$OFF" "$LR" <<'PYCHK'
import json, os, sys
out, arm, off, LR = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
audit = None
for line in open(os.path.join(out, "srd.jsonl")):
    try: r = json.loads(line)
    except Exception: continue
    if r.get("type") == "barrier_audit":
        audit = r
assert audit is not None, "no barrier_audit row -- the patched exit flush did not run"
assert audit["barrier_center"] == arm, f"audit says {audit['barrier_center']}, arm is {arm}"
print(f"V28_AUDIT arm={arm} off={off} steps={audit['steps']} eef_fire={audit['eef_fire']} "
      f"obj_fire={audit['obj_fire']} obj_gated_out={audit['obj_gated_out']} "
      f"obj_unavailable={audit['obj_unavailable']}")
if arm == "eef":
    assert audit["obj_fire"] == 0, "eef arm must never run the object barrier"
if arm == "object":
    assert audit["eef_fire"] == 0, "object arm must never run the gripper barrier"

res = json.load(open(os.path.join(out, "result.json")))
md = res.get("metric_decomposition", {})
print(f"V28_RESULT arm={arm} off={off} sr={res.get('successes')} polcc={md.get('policy_induced_cc')}")

# ---- regression gate: the default path must be unchanged by the patch ----
if arm == "eef":
    ref = None
    for cand in (f"{LR}/v24_shbridge/shield/off{off}/result.json",
                 f"{LR}/v9_power/shield/off{off}/result.json"):
        if os.path.exists(cand):
            ref = json.load(open(cand)); break
    if ref is None:
        print(f"V28_REGRESSION_NO_REFERENCE off={off}")
    else:
        rmd = ref.get("metric_decomposition", {})
        ok = (int(ref.get("successes", -1)) == int(res.get("successes", -2))
              and rmd.get("policy_induced_cc") == md.get("policy_induced_cc"))
        print(f"V28_REGRESSION off={off} ref_sr={ref.get('successes')} new_sr={res.get('successes')} "
              f"ref_cc={rmd.get('policy_induced_cc')} new_cc={md.get('policy_induced_cc')} "
              f"{'MATCH' if ok else 'MISMATCH'}")
        assert ok, "patched eef path no longer reproduces the historical result"
print("V28_CELL_VERIFIED")
PYCHK
