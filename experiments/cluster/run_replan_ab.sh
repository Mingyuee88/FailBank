#!/usr/bin/env bash
#$ -N ReplanAB
#$ -cwd
#$ -q gpu@HOST_B
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-64
#$ -tc 4
#$ -j y
#
# Does replan_steps explain why our CC sits far below the Arena leaderboard?
#
# Every number we have was produced with --replan-steps 1: re-infer every step, execute only
# the first action of the chunk. Arena's own evaluation yaml ships replan_steps: 5, which
# runs 5 steps open-loop per inference. Open-loop drift is exactly the regime where a policy
# clips an obstacle, so replan=1 is plausibly the most conservative setting available and
# would depress CC across the board.
#
# Two suites, base policy both times, 16 offsets each, replan 1 vs 5 -- 64 cells:
#   safety_dynamic_obstacles level_2 task 0  (the dynamic arm's task)
#   safety_static_obstacles  level_1 task 2  (the static arm's task)
#
# PINNED to l40s-005: the whole point is a within-host contrast.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
OUTROOT=${WORK_ROOT}/lora_dagger/replan_ab

i=$((SGE_TASK_ID - 1))
S=$((i / 32)); R=$((i % 32)); P=$((R / 16)); OFF=$((R % 16))
case "$S" in
  0) SUITE=safety_dynamic_obstacles; LV=2; TID=0; SN=dyn ;;
  1) SUITE=safety_static_obstacles;  LV=1; TID=2; SN=sta ;;
esac
case "$P" in
  0) RP=1 ;;
  1) RP=5 ;;
esac
CELL="${SN}_rp${RP}_off${OFF}"
OUT=$OUTROOT/$CELL
mkdir -p "$OUT"
if [ -s "$OUT/result.json" ]; then echo "SKIP $CELL"; exit 0; fi
hostname > "$OUT/host.txt"

unset SE_VLA_POLICY_CHECKPOINT_DIR || true    # base policy on both arms
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SHIELD_OBSERVE_ONLY=1
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="replanab_${CELL}"

for v in SE_VLA_SEED SE_VLA_RESULT_PATH SE_VLA_SRD_NEAR_DISTANCE_M \
         SE_VLA_SRD_RELEASE_DISTANCE_M SE_VLA_SRD_MIN_CLOSING_M_PER_STEP; do
  eval "test -n \"\${$v:-}\"" || { echo "MISSING_REQUIRED_ENV $v"; exit 1; }
done

echo "=== ReplanAB $CELL suite=$SUITE L$LV t$TID replan=$RP on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite "$SUITE" \
  --task-level "$LV" --task-id "$TID" --offset "$OFF" --seed 17 --replan-steps "$RP" \
  --trials 1 --arm base \
  --port-base $((49000 + 20 * (SGE_TASK_ID - 1))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "REPLANAB_EXIT=$rc cell=$CELL"
[ -s "$OUT/result.json" ] || { echo "REPLANAB_CELL_FAILED $CELL"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/result.json" "$CELL" "$RP" <<'PYCHK'
import json, sys
r = json.load(open(sys.argv[1])); md = r.get("metric_decomposition") or {}
got = r.get("replan_steps")
assert str(got) == sys.argv[3], f"replan_steps not applied: asked {sys.argv[3]}, report says {got}"
print("REPLANAB cell=%s replan=%s sr=%s offcc=%s polcc=%s" % (
    sys.argv[2], got, r.get("successes"), md.get("official_cc"), md.get("policy_induced_cc")))
PYCHK
