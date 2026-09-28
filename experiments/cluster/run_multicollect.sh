#!/usr/bin/env bash
#$ -N MCol
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-47
#$ -tc 4
#$ -j y
#
# CROSS-TASK COLLECTION for the visual distance head.
#
# Every existing bank -- c1_collect, r2_collect, r3_collect, phase1_collection, v16, v26,
# v36 -- records the SAME task (mango / L1t2). A head trained on them has seen one layout,
# which is exactly why the proprioceptive version worked on L1-T2 (SR 1.0 -> 64.0) and
# scored 0 on L1-T1 and L1-T4. More capacity will not fix that; more layouts will.
#
# So this collects (observation, oracle-distance) pairs on the other tasks. The policy used
# is the current best, R2_q00 (no staging, quiet 0: SR 90.8, CC 18.5), so the recorded state
# distribution matches what the head will actually see at deployment.
#
# The shield stays OBSERVE-ONLY: with it steering, the trajectory never approaches the
# hazard and the near-contact band -- the only part that matters for a trigger -- is never
# recorded. Both that and the per-step cost-pair geometry are asserted per cell.
set -euo pipefail
: "${TID:?set TID via qsub -v}"
: "${TAG:?set TAG via qsub -v}"
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/mcol_$TAG
CK=$LR/nocurr/ckpt/R2_q00/offset_0
[ -d "$CK/params" ] || { echo "MC_ABORT: missing policy at $CK"; exit 1; }
OFFS=(0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47)
i=$((SGE_TASK_ID - 1))
[ "$i" -lt "${#OFFS[@]}" ] || { echo "no cell"; exit 0; }
OFF=${OFFS[$i]}
OUT=$OUTROOT/off$OFF
mkdir -p "$OUT" "$OUTROOT/batch"
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_POLICY_CHECKPOINT_DIR="$CK"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SHIELD_OBSERVE_ONLY=1
export SE_VLA_PHASE1_RECORD=1 SE_VLA_PHASE1_CANDIDATES=1
export SE_VLA_PHASE1_ROOT=$OUTROOT/records
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="mcol${TAG}_off${OFF}"
echo "=== MCol $TAG(tid=$TID) off=$OFF policy=R2_q00 on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id "$TID" --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((44000 + 20 * (SGE_TASK_ID - 1))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "MC_EXIT=$rc off=$OFF"
[ -s "$OUT/result.json" ] || { echo "MC_CELL_FAILED off=$OFF"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUTROOT/records" "$OFF" "$OUT" "$TAG" <<'PYCHK'
import json, pathlib, sys
root, off, out, tag = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4]
res = json.load(open(out / "result.json")); md = res.get("metric_decomposition") or {}
print(f"MC_RESULT off={off} sr={res.get('successes')} polcc={md.get('policy_induced_cc')}")
want = f"mcol{tag}_off{off}"
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
print("MC_CELL_VERIFIED")
PYCHK
