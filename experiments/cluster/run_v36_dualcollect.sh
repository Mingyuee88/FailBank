#!/usr/bin/env bash
#$ -N v36_dualc
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 2
#$ -t 1-29
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v36_dualcollect/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v36_dualcollect/batch/j.$JOB_ID.$TASK_ID.out
#
# v36_dualcollect -- close the loop between the diagnosis and the method.
#
# The diagnosis says the published shield brakes for a cost that cannot occur: on the
# development arena every offset it destroyed had base policy-induced cost <= 7 while
# every offset it repaired had >= 225 (AUC 0.000, p=0.00024), and its first projection
# lands during the reach in 36 of 36 episodes with only the already-true `fall` conjunct
# holding. Re-centring the barrier on the manipulated object and running both barriers in
# sequence gives a teacher that is not worse on any axis: repair 11/11 vs 10/11,
# retention 29/36 vs 27/36, net 40/47 vs 37/47, cost still down 82%.
#
# So far that is a better SHIELD. The method's claim is about what a policy inherits from
# a teacher, and nothing yet connects the two. This collects the same teacher signal from
# the better teacher.
#
# WHY THIS IS A CLEAN COMPARISON. It is v29_basedist with exactly one thing changed. Same
# 29 training-half offsets, same base policy, same recorder, same host, same shield gains
# and radii -- only SE_VLA_AEGIS_BARRIER_CENTER goes from the default `eef` to `dual`.
# Training then aggregates round-1 records with these, exactly as R2CTRL aggregated
# round-1 records with v29_basedist. R2CTRL and the arm trained from here therefore differ
# ONLY in the barrier centre of the teacher that produced half their data, which makes
# "does a better teacher produce a better student" a one-variable question.
#
# The object term is gated on the object riding with the gripper, and the job asserts the
# object barrier actually had an object to centre on: on a suite whose cost predicate the
# feature extractor cannot parse, `dual` silently degrades to `eef` and would produce a
# null meaning "the knob did nothing" rather than "the better teacher does not help".
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger


# 47 usable L1t2 offsets minus v18_split heldout_success_offsets
TRAIN_OFFS=(0 1 3 5 6 8 10 12 14 16 18 20 22 24 26 27 28 29 31 33 35 36 37 39 41 42 43 45 47)
OFF=${TRAIN_OFFS[$((SGE_TASK_ID - 1))]:-}
[ -n "$OFF" ] || { echo "no offset for task $SGE_TASK_ID"; exit 0; }

OUT=$LR/v36_dualcollect/off$OFF
RECROOT=$LR/v36_dualcollect/records
mkdir -p "$OUT" "$RECROOT" $LR/v36_dualcollect/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true
# NB: pipefail is on, and `find` on a not-yet-created blobs dir exits nonzero, which
# would kill the job before it ever starts. Swallow that specific failure.
BLOBS_BEFORE=$( { find "$RECROOT/blobs" -type f 2>/dev/null || true; } | wc -l )

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
# the policy under correction is BASE -- same distribution as round 1 and as v29_basedist
unset SE_VLA_POLICY_CHECKPOINT_DIR || true
# THE one variable: the teacher's barrier centre
export SE_VLA_AEGIS_BARRIER_CENTER=dual
export SE_VLA_AEGIS_GRASP_GATE_M=0.12
# identical shield recipe to round 1 so the two rounds' teacher signals are comparable
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_PHASE1_RECORD=1 SE_VLA_PHASE1_CANDIDATES=1
export SE_VLA_PHASE1_ROOT="$RECROOT"
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="dualc_off${OFF}"

echo "=== v36 dual-barrier teacher collection off=$OFF on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((62000 + 20 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "DUALC_EXIT=$rc off=$OFF"

# ---- assertion 1: the STOCK checkpoint, not a folded LoRA, is what the server restored
if grep -q "Restoring checkpoint from ${WORK_ROOT}/lora_dagger/v20_fold" \
     "$OUT"/server_port*.log 2>/dev/null; then
  echo "BASEDIST_CONTAMINATED off=$OFF -- a folded LoRA checkpoint leaked into the control"
  exit 1
fi
echo "BASE_CHECKPOINT_VERIFIED off=$OFF"

# ---- assertions 2 and 3: shield really ran, recorder really wrote
BLOBS_AFTER=$( { find "$RECROOT/blobs" -type f 2>/dev/null || true; } | wc -l )
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" <<'PYAUDIT'
import json, os, sys
out = sys.argv[1]
audit = None
for line in open(os.path.join(out, "srd.jsonl")):
    try: r = json.loads(line)
    except Exception: continue
    if r.get("type") == "barrier_audit":
        audit = r
assert audit is not None, "no barrier_audit row"
assert audit["barrier_center"] == "dual", f"barrier centre is {audit['barrier_center']}"
steps = max(int(audit["steps"]), 1)
print(f"V36_AUDIT steps={steps} eef_fire={audit['eef_fire']} obj_fire={audit['obj_fire']} "
      f"obj_gated_out={audit['obj_gated_out']} obj_unavailable={audit['obj_unavailable']}")
assert audit["obj_unavailable"] / steps < 0.5, (
    f"object barrier had no object on {audit['obj_unavailable']}/{steps} steps -- "
    f"this teacher is INAPPLICABLE here, not null")
print("V36_BARRIER_VERIFIED")
PYAUDIT
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/result.json" "$OFF" "$BLOBS_BEFORE" "$BLOBS_AFTER" <<'PYCHK'
import json, sys
r = json.load(open(sys.argv[1])); off, before, after = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ad = r.get("adapter", {})
md = r.get("metric_decomposition", {})
print(f"DUALC off={off} shield_corrections={ad.get('corrections_applied')} "
      f"sr={r.get('successes')} polcc={md.get('policy_induced_cc')} blobs {before}->{after}")
assert r.get("status") == "pass", f"episode status {r.get('status')}"
# the shield being silent on an offset R1 already handles is expected and fine; what is
# not fine is the recorder producing nothing anywhere, so growth is asserted globally by
# the collector's summary rather than per cell. Here only assert the run was well-formed.
assert ad.get("physical_units_gate") == "pass", "physical units gate"
print("DUALC_CELL_VERIFIED")
PYCHK
