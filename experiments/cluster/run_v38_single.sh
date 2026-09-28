#!/usr/bin/env bash
#$ -N v38_single
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -pe smp 2
#$ -t 1-150
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v38_single/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v38_single/batch/j.$JOB_ID.$TASK_ID.out
#
# v38_single -- how much of our shield's damage is just "it constrains too many things"?
#
# Published AEGIS asks a VLM for exactly ONE object most likely to obstruct, and builds
# its barrier on that. Our reimplementation builds a barrier on EVERY object named in the
# BDDL cost predicate, from ground-truth positions, and fires on 20-47% of steps -- while
# in the paper (Supplementary Table S1, SafeLIBERO-Spatial, translational-only) AEGIS
# RAISES task success from 59.8 to 73.3. Ours lowers it. Something is different, and the
# candidate differences are: how many obstacles, ellipsoid vs sphere, QP vs closed-form
# projection, barrier centre, class-K gain, and perception vs oracle.
#
# This job isolates exactly one of them: the number of obstacles. Ground-truth geometry is
# deliberately kept, so nothing else moves.
#
#   MODE=multi   every cost-state object      -- the current behaviour, run as a
#                                                REGRESSION GATE against the historical
#                                                cells; must reproduce them exactly
#   MODE=single  only the hazard named in the active cost pair
#
# READING, fixed before the run:
#   * if `single` stops destroying base successes -> lack of selectivity is the cause of
#     our variant's damage, and the finding is about our reimplementation, not about
#     safety filters in general
#   * if `single` destroys them anyway -> the cause lies in barrier geometry or the
#     solver, and only the faithful SafeLIBERO reproduction can settle it
#
# Either way this does not replace the reproduction. It is the cheap diagnostic that runs
# while the pi0.5-libero and GroundingDINO downloads are in flight.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

: "${MODE:?MODE must be passed with qsub -v MODE=single|multi}"
: "${DOMAIN:?DOMAIN must be passed with qsub -v DOMAIN=dev|xfer}"
i=$((SGE_TASK_ID - 1))
if [ "$DOMAIN" = "dev" ]; then
  OFFS=(0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47)
  [ "$i" -lt "${#OFFS[@]}" ] || { echo "no cell for task $SGE_TASK_ID"; exit 0; }
  TID=2; OFF=${OFFS[$i]}
else
  TASKS=(1 3 4); TID=${TASKS[$((i / 50))]}; OFF=$((i % 50))
fi

OUT=$LR/v38_single/$MODE/$DOMAIN/L1t${TID}/off${OFF}
mkdir -p "$OUT" $LR/v38_single/batch
if [ -s "$OUT/result.json" ] && \
   external/VLA-Arena/envs/openpi/.venv/bin/python -c \
     'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="pass" else 1)' \
     "$OUT/result.json" 2>/dev/null; then
  echo "SKIP $OUT"; exit 0
fi
rm -f "$OUT/result.json"
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
unset SE_VLA_POLICY_CHECKPOINT_DIR || true
# identical shield recipe to every earlier measurement; only the obstacle set changes
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_AEGIS_BARRIER_CENTER=eef
export SE_VLA_AEGIS_SINGLE_OBSTACLE=$([ "$MODE" = "single" ] && echo 1 || echo 0)
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="v38_${MODE}_${DOMAIN}_L1t${TID}_off${OFF}"

echo "=== v38 mode=$MODE $DOMAIN L1t${TID} off=$OFF on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id "$TID" --offset "$OFF" --seed 17 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((62000 + 20 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "V38_EXIT=$rc mode=$MODE $DOMAIN L1t$TID off=$OFF"

if [ ! -s "$OUT/result.json" ] || ! external/VLA-Arena/envs/openpi/.venv/bin/python -c \
     'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="pass" else 1)' \
     "$OUT/result.json" 2>/dev/null; then
  echo "V38_CELL_FAILED mode=$MODE $DOMAIN L1t$TID off=$OFF -- removing for retry"
  rm -f "$OUT/result.json"; exit 1
fi

# The audit counts obstacles from the code that used them, never from the env var. In
# `single` mode the mean must be at most 1; in `multi` mode it must exceed 1 somewhere, or
# the two modes are the same run and the comparison is empty.
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$MODE" "$DOMAIN" "$TID" "$OFF" "$LR" <<'PYCHK'
import json, os, sys
out, mode, dom, tid, off, LR = sys.argv[1:7]
audit = None
for line in open(os.path.join(out, "srd.jsonl")):
    try: r = json.loads(line)
    except Exception: continue
    if r.get("type") == "barrier_audit":
        audit = r
assert audit is not None, "no barrier_audit row"
steps = max(int(audit.get("obstacle_steps", 0)), 1)
mean_obs = audit.get("obstacle_count_sum", 0) / steps
r = json.load(open(os.path.join(out, "result.json")))
md = r.get("metric_decomposition", {})
print(f"V38_RESULT mode={mode} {dom} L1t{tid} off{off} sr={r.get('successes')} "
      f"polcc={md.get('policy_induced_cc')} mean_obstacles={mean_obs:.2f} "
      f"eef_fire={audit['eef_fire']} steps={audit['steps']}")
if mode == "single":
    assert mean_obs <= 1.0 + 1e-9, f"single mode still saw {mean_obs:.2f} obstacles per step"

# regression gate: multi mode must reproduce the historical cells exactly, or the patch
# changed the default path and every comparison against the old numbers is void
if mode == "multi" and dom == "dev":
    ref = None
    for cand in (f"{LR}/v28_barrier/eef/off{off}/result.json",
                 f"{LR}/v24_shbridge/shield/off{off}/result.json",
                 f"{LR}/v9_power/shield/off{off}/result.json"):
        if os.path.exists(cand):
            ref = json.load(open(cand)); break
    if ref is None:
        print(f"V38_REGRESSION_NO_REFERENCE off={off}")
    else:
        rmd = ref.get("metric_decomposition", {})
        ok = (int(ref.get("successes", -1)) == int(r.get("successes", -2))
              and rmd.get("policy_induced_cc") == md.get("policy_induced_cc"))
        print(f"V38_REGRESSION off={off} ref_sr={ref.get('successes')} new_sr={r.get('successes')} "
              f"{'MATCH' if ok else 'MISMATCH'}")
        assert ok, "patched multi path no longer reproduces the historical result"
print("V38_CELL_VERIFIED")
PYCHK
