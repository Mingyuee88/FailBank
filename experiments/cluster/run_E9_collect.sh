#!/usr/bin/env bash
#$ -N E9_collect
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_D
#$ -pe smp 4
#$ -t 1-97
#$ -tc 3
#$ -o ${WORK_ROOT}/E9_collect/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/E9_collect/batch/j.$JOB_ID.$TASK_ID.out
#
# E9 -- teacher collection with the PUBLISHED AEGIS shield, replacing the local sphere-CBF
# teacher every distilled arm so far was trained from.
#
# WHY. The paper wants to say "we distil a published safety filter". What was actually
# distilled is our own oracle-CBF controller: run_phase1_collect.sh passes alpha /
# eef_radius / oracle_radius / margin, and SE_VLA_SHIELD_IMPL defaults to "local". The
# env vars are named SE_VLA_AEGIS_* for historical reasons, which is how the confusion
# survived. This arm fixes the provenance.
#
# COLLECTION SET. 97 unique offsets: L1t2 (47) + L1t4 (50). Deliberately DISJOINT from the
# preregistered confirmatory endpoint, which is the transfer domain L1t1+L1t3.
#
# ONE seed per offset, not two. --seed is a verified no-op in this harness -- evaluation is
# deterministic on a pinned host and GPU model is the only variance source -- so a second
# seed re-runs the same episode and buys nothing. The previous teacher used 10 offsets x 2
# seeds; the binding constraint on attribution power is unique SITUATIONS, not steps, so
# this trades duplicate seeds for 9.7x the offset diversity at similar cost.
#
# RESIDUAL DEFINITION. run.py records the translational correction only. Under the
# translational variant a naive 7-vector residual has rotation coordinates identically
# equal to -nominal_rotation at every step: a hard-coded channel clamp, not a barrier-
# derived direction. Distilling it would let a student mimic AEGIS by learning to stop
# rotating and would contaminate the TRUE-vs-null contrast. The clamp is recorded
# separately as vlsa_residual_split for a factorial arm.
#
# PERCEPTION. Local GLM-4.5V, fail-closed. E7 measured the local/hosted difference on the
# dev stratum: polCC identical on 47/47, success on 46/47. Every arm trained from these
# records will therefore share one perception backend, which is the property that matters.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
OUTROOT=${WORK_ROOT}/E9_collect

: "${SE_VLA_GLM_BASE_URL:=http://HOST_E:8502/v1}"
export SE_VLA_GLM_BASE_URL
if ! curl -sf --max-time 20 "${SE_VLA_GLM_BASE_URL%/v1}/v1/models" | grep -q 'glm-4.5v'; then
  echo "E9_ABORT: local GLM not serving glm-4.5v at $SE_VLA_GLM_BASE_URL"; exit 1
fi
unset ZHIPUAI_API_KEY || true

# task 1..47 -> L1t2 offsets; task 48..97 -> L1t4 offsets 0..49
T2=(0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47)
i=$((SGE_TASK_ID - 1))
if [ "$i" -lt "${#T2[@]}" ]; then
  TID=2; OFF=${T2[$i]}
else
  TID=4; OFF=$((i - ${#T2[@]}))
  [ "$OFF" -lt 50 ] || { echo "no cell for task $SGE_TASK_ID"; exit 0; }
fi

OUT=$OUTROOT/L1t${TID}/off${OFF}
mkdir -p "$OUT" "$OUTROOT/batch"
if [ -s "$OUT/result.json" ] && \
   external/VLA-Arena/envs/openpi/.venv/bin/python -c \
     'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="pass" else 1)' \
     "$OUT/result.json" 2>/dev/null; then echo "SKIP $OUT"; exit 0; fi
rm -f "$OUT/result.json" "$OUT/vlsa_steps.jsonl"
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
unset SE_VLA_POLICY_CHECKPOINT_DIR || true      # teacher runs over the BASE policy
export SE_VLA_SHIELD_IMPL=vlsa_aegis
export SE_VLA_VLSA_DOF=3
export SE_VLA_VLSA_STEP_TRACE=1 SE_VLA_VLSA_STEPS_PATH="$OUT/vlsa_steps.jsonl"
export SE_VLA_VLSA_RESIDUAL=translation
export SE_VLA_ARENA_SUITE=safety_static_obstacles
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
# the phase-1 recorder, same switches run_phase1_collect.sh used
export SE_VLA_PHASE1_RECORD=1 SE_VLA_PHASE1_CANDIDATES=1
export SE_VLA_PHASE1_ROOT=$OUTROOT/records
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="e9_L1t${TID}_off${OFF}"

echo "=== E9 AEGIS-teacher collect L1t${TID} off=$OFF on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id "$TID" --offset "$OFF" --seed 17 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((30000 + 20 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "E9_EXIT=$rc L1t$TID off=$OFF"

if [ ! -s "$OUT/result.json" ] || ! external/VLA-Arena/envs/openpi/.venv/bin/python -c \
     'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="pass" else 1)' \
     "$OUT/result.json" 2>/dev/null; then
  echo "E9_CELL_FAILED L1t$TID off=$OFF -- removing for retry"; rm -f "$OUT/result.json"; exit 1
fi

# The whole point of this arm is the RESIDUAL. A cell that ran the shield but wrote no
# override candidates is useless as teacher data and must not look successful.
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$TID" "$OFF" <<'PYCHK'
import json, os, sys
out, tid, off = sys.argv[1], sys.argv[2], sys.argv[3]
audit = None
nover = 0
nsplit = 0
for line in open(os.path.join(out, "srd.jsonl")):
    try: r = json.loads(line)
    except Exception: continue
    t = r.get("type")
    if t == "vlsa_audit": audit = r
    elif t == "override_candidate" and r.get("provenance") == "vlsa_aegis_override": nover += 1
    elif t == "vlsa_residual_split": nsplit += 1
assert audit is not None, "no vlsa_audit row -- the published shield never ran"
assert audit.get("dof") == 3, f"cell ran dof={audit.get('dof')}, expected 3"
r = json.load(open(os.path.join(out, "result.json")))
md = r.get("metric_decomposition", {})
print(f"E9_RESULT L1t{tid} off={off} sr={r.get('successes')} polcc={md.get('policy_induced_cc')} "
      f"obstacle={audit['obstacle_name']!r} steps={audit['steps']} "
      f"active={audit.get('constraint_active_steps')} override_sum={audit['override_norm_sum']:.4f} "
      f"override_candidates={nover} residual_split_rows={nsplit}")
assert audit["perception_ok"], "perception produced no ellipsoid"
name = str(audit["obstacle_name"])
assert name and len(name) <= 40 and "\n" not in name, f"obstacle name looks wrong: {name[:200]!r}"
assert audit["override_norm_sum"] > 0, "the shield never changed an action"
assert nover > 0, "shield ran but buffered ZERO override candidates -- no teacher signal"
assert nsplit > 0, "no residual-split telemetry -- translation/rotation not separated"
print("E9_CELL_VERIFIED")
PYCHK
