#!/usr/bin/env bash
#$ -N R2_collect
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 4
#$ -t 1-47
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/r2_collect/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/r2_collect/batch/j.$JOB_ID.$TASK_ID.out
#
# ROUND 2 COLLECTION -- the first step of the loop that has never actually run.
#
# Round 1 was: collect on D -> train pi_1 (CURRICULUM). That is one DAgger iteration, not
# self-evolution. Self-evolution requires the next round to collect on the CURRENT policy's
# own state distribution, which is what this does: pi_1 = CURRICULUM_p2 drives, the shield
# runs OBSERVE-ONLY (computes and logs its correction without steering), and every step is
# recorded with the cost-pair geometry needed to stage the curriculum again.
#
# The bank accumulates: round-2 records are added to round-1's, they do not replace them.
# Staging is recomputed for the new policy, because which corrections count as "early and
# learnable" depends on the policy being trained.
#
# Target to beat, fixed from the published paper and not re-measured:
#   AEGIS  SR 35/47   crash 1   safe-fail 11   polCC 199
# Winning means crash <= 1 AND SR > 35 AND safe-fail < 11 -- AEGIS buys its low cost with 11
# episodes that touch nothing and complete nothing, which is explicitly not acceptable here.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/r2_collect
CK=$LR/se_curr/ckpt/CURRICULUM_p2/offset_0
[ -d "$CK/params" ] || { echo "R2_ABORT: missing pi_1 at $CK"; exit 1; }
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
export SE_VLA_EPISODE_ID="r2_pi1_off${OFF}"
echo "=== R2 collect off=$OFF policy=pi_1(CURRICULUM) on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((44000 + 20 * (SGE_TASK_ID - 1))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "R2_EXIT=$rc off=$OFF"
[ -s "$OUT/result.json" ] || { echo "R2_CELL_FAILED off=$OFF"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUTROOT/records" "$OFF" "$OUT" <<'PYCHK'
import json, pathlib, sys
root, off, out = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
res = json.load(open(out / "result.json")); md = res.get("metric_decomposition") or {}
print(f"R2_RESULT off={off} sr={res.get('successes')} polcc={md.get('policy_induced_cc')}")
want = f"r2_pi1_off{off}"
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
print("R2_CELL_VERIFIED")
PYCHK
