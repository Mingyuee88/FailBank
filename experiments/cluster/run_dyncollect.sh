#!/usr/bin/env bash
#$ -N DynCol
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-47
#$ -tc 4
#$ -j y
#
# ROUND-1 COLLECTION ON safety_dynamic_obstacles.
#
# Everything trained so far comes from static obstacles. Dynamic has the same cost structure
# (Fall + InContact + CheckGripperContact) so the semantics dispatch already covers it, but
# the hazards move, which the barrier geometry never sees.
#
# Expect a thin harvest and say so up front: on dyn_d0 the base policy left only 6% of cells
# carrying policy-induced cost (the rest is reset contamination or clean), against 46-94% on
# the static tasks. Task 0 is the only pick-and-place in this suite; the other four are push
# tasks whose cost predicate includes Fall on teapots/mugs that topple during reset.
#
# Collected with the BASE policy, which is where a first round starts and which fails most,
# so the harvest is as large as this suite allows.
set -euo pipefail
: "${TID:?set TID via qsub -v}"
: "${TAG:?set TAG via qsub -v}"
: "${TID:?set TID via qsub -v}"
: "${TAG:?set TAG via qsub -v}"
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/dcol_$TAG
CK=

OFFS=(0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47)
i=$((SGE_TASK_ID - 1))
[ "$i" -lt "${#OFFS[@]}" ] || { echo "no cell"; exit 0; }
OFF=${OFFS[$i]}
OUT=$OUTROOT/off$OFF
mkdir -p "$OUT" "$OUTROOT/batch"
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
if [ -n "${POLICY_CK:-}" ]; then
  [ -d "$POLICY_CK/params" ] || { echo "DC_ABORT: no checkpoint at $POLICY_CK"; exit 1; }
  export SE_VLA_POLICY_CHECKPOINT_DIR="$POLICY_CK"
else
  unset SE_VLA_POLICY_CHECKPOINT_DIR || true   # base policy, round 1
fi
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SHIELD_OBSERVE_ONLY=1
# Hazard geometry lookahead. Unset -> 0.0 -> the original snapshot behaviour, verified
# byte-identical against the pre-patch function. >0 advances each hazard along its own
# world-frame velocity before the sphere is built, so the barrier normal points at where
# the hazard WILL be rather than where it is. Only the geometry changes; the CBF projection
# and the training recipe never see it.
export SE_VLA_HAZARD_LOOKAHEAD_S=${TAU:-0.0}
export SE_VLA_PHASE1_RECORD=1 SE_VLA_PHASE1_CANDIDATES=1
export SE_VLA_PHASE1_ROOT=$OUTROOT/records
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="dcol${TAG}_off${OFF}"
for v in SE_VLA_SEED SE_VLA_RESULT_PATH SE_VLA_SRD_NEAR_DISTANCE_M \
         SE_VLA_SRD_RELEASE_DISTANCE_M SE_VLA_SRD_MIN_CLOSING_M_PER_STEP; do
  eval "test -n \"\${$v:-}\"" || { echo "MISSING_REQUIRED_ENV $v"; exit 1; }
done
echo "=== MCol $TAG(level=${LEVEL:-1} tid=$TID) off=$OFF tau=$SE_VLA_HAZARD_LOOKAHEAD_S policy=${POLICY_CK:-base} on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_dynamic_obstacles \
  --task-level "${LEVEL:-1}" --task-id "$TID" --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((44000 + 20 * (SGE_TASK_ID - 1))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "DC_EXIT=$rc off=$OFF"
[ -s "$OUT/result.json" ] || { echo "DC_CELL_FAILED off=$OFF"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUTROOT/records" "$OFF" "$OUT" "$TAG" <<'PYCHK'
import json, pathlib, sys
root, off, out, tag = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4]
res = json.load(open(out / "result.json")); md = res.get("metric_decomposition") or {}
print(f"DC_RESULT off={off} sr={res.get('successes')} polcc={md.get('policy_induced_cc')}")
want = f"dcol{tag}_off{off}"
ep = None
for meta in root.glob("episodes/*/*/episode.json"):
    try:
        if json.load(meta.open()).get("task_description") == want:
            ep = meta.parent; break
    except Exception:
        continue
assert ep is not None, f"no episode recorded for {want}"
n = have = 0
for line in (ep / "raw_steps.jsonl").open():
    r = json.loads(line); ri = r.get("runtime_info") or {}
    n += 1
    have += int(ri.get("cost_pair_min_distance") is not None)
    assert ri.get("shield_observe_only") is True, "shield was steering; the collection is void"
print(f"  episode={ep.parent.name}/{ep.name[:8]} steps={n} with_cost_pair={have}")
assert have == n and n > 0, f"cost-pair geometry missing on {n-have}/{n} steps"
print("DC_CELL_VERIFIED")
PYCHK
