#!/usr/bin/env bash
#$ -N R3_collect
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 4
#$ -t 1-47
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/r3_collect/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/r3_collect/batch/j.$JOB_ID.$TASK_ID.out
#
# ROUND 3 COLLECTION -- extends the self-evolution chain to a third point.
#
# Rounds 1 and 2 are measured (L1t2, 50 offsets x 5 repeats, clustered by offset):
#     base  SR 35.0 +-3.29   crash 14.6   polCC 3924
#     r1    SR 38.2 +-2.32   crash 11.6   polCC 3002    base->r1  dSR +0.064  p=0.043
#     r2    SR 42.6 +-1.62   crash  7.4   polCC 1987    r1->r2    dSR +0.088  p=0.016
# The gain did not decay -- r1->r2 is LARGER than base->r1 -- because round 2 trains on the
# failures round 1's own policy produced, which are the harder ones by construction. A third
# point decides whether that is a trend or a two-point coincidence, and it is the difference
# between a claim and a curve in the paper.
#
# So: r2 drives, the shield runs OBSERVE-ONLY (computes and logs its correction without
# steering -- with it in the loop the trajectory never approaches the hazard and no failure
# is recorded at all), and every step carries the cost-pair geometry needed to stage again.
# The per-cell verifier asserts both, and refuses the cell otherwise.
#
# The bank accumulates: round-3 records add to rounds 1-2, they do not replace them.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/r3_collect
CK=$LR/se_curr2/ckpt/CURRICULUM_p2/offset_0
[ -d "$CK/params" ] || { echo "R3_ABORT: missing r2 policy at $CK"; exit 1; }
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
export SE_VLA_EPISODE_ID="r3_pi2_off${OFF}"
echo "=== R3 collect off=$OFF policy=r2(se_curr2/CURRICULUM_p2) on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((44000 + 20 * (SGE_TASK_ID - 1))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "R3_EXIT=$rc off=$OFF"
[ -s "$OUT/result.json" ] || { echo "R3_CELL_FAILED off=$OFF"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUTROOT/records" "$OFF" "$OUT" <<'PYCHK'
import json, pathlib, sys
root, off, out = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
res = json.load(open(out / "result.json")); md = res.get("metric_decomposition") or {}
print(f"R3_RESULT off={off} sr={res.get('successes')} polcc={md.get('policy_induced_cc')}")
want = f"r3_pi2_off{off}"
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
print("R3_CELL_VERIFIED")
PYCHK
