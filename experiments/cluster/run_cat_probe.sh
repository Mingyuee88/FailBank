#!/usr/bin/env bash
#$ -N CatProbe
#$ -cwd
#$ -q gpu@HOST_D
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-600
#$ -tc 4
#$ -j y
#
# Does the shield's failure generalise beyond the two obstacle suites?
#
# We have only ever evaluated safety_static_obstacles and safety_dynamic_obstacles. The Safety
# category has five suites, all with the same level_0/1/2 x 5-task grid. This probes the three
# we have never run, at level_2 (2 hazards, the hardest), with the two TRAINING-FREE arms:
#
#   base   -- pi0.5-ft, shield observe-only (computes, never steers)
#   aegis  -- pi0.5-ft, shield in the loop (vlsa_aegis + GLM-4.5V), actually steers
#
#   3 suites x 5 tasks x 20 offsets x 2 arms = 600 cells
#
# `ours` is deliberately absent: it needs a collect+distil round per suite, which AEGIS does
# not pay. Adding a static-obstacles adapter here would compare a TRANSFERRED ours against a
# NATIVE aegis. Stage 2 runs the loop natively on whichever suites show a failure here.
#
# This doubles as the headroom probe demanded by the "count headroom before evaluating" rule:
# a suite whose base is at ceiling or on the floor cannot carry a motivation claim.
#
# PINNED to a6k-002 (GPU model alone moves SR by up to 29 points, see §0d). GLM serves from
# a6k-003 over HTTP, which does not affect the evaluation host.
#
# Recipe derived verbatim from run_tau_eval.sh (base arm) and run_aegis_t3.sh (aegis arm);
# only the suite/task/offset indexing and the output root are new.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/cat_probe

i=$((SGE_TASK_ID - 1))
ARM_I=$((i / 300)); j=$((i % 300))
SUITE_I=$((j / 100)); k=$((j % 100))
TID=$((k / 20)); OFF=$((k % 20))

case "$SUITE_I" in
  0) SUITE=safety_cautious_grasp;     SHORT=grasp ;;
  1) SUITE=safety_hazard_avoidance;   SHORT=hazard ;;
  2) SUITE=safety_state_preservation; SHORT=state ;;
esac
case "$ARM_I" in
  0) ARM=base ;;
  1) ARM=aegis ;;
esac

CELL="${SHORT}_t${TID}_${ARM}_off${OFF}"
OUT=$OUTROOT/$CELL
mkdir -p "$OUT"
if [ -s "$OUT/result.json" ]; then echo "SKIP $CELL"; exit 0; fi
hostname > "$OUT/host.txt"
# run.py is always called with --arm base; the arm lives in env vars, so record it here.
# result.json's own "arm" field is NOT trustworthy (see RESULTS_FOR_PAPER.md §0b).
printf '%s\n' "$ARM" > "$OUT/arm.txt"
printf '%s %s %s %s\n' "$SUITE" "$TID" "$OFF" "$ARM" > "$OUT/cell_id.txt"

# ---- arm-specific shield configuration ----
unset SE_VLA_POLICY_CHECKPOINT_DIR || true          # no adapter in either arm
if [ "$ARM" = "aegis" ]; then
  # vlsa_port reads the suite from THIS env var; leaving it unset makes _crop_suite_for see
  # '' and abort. run_aegis_t3.sh omits it, which is why all 18 of its aegis cells failed.
  export SE_VLA_ARENA_SUITE="$SUITE"
  # only suites with a MEASURED crop box may run the shield (see probe_crop_newsuites.py);
  # borrowing a box would give an empty cloud and a fake "no effect".
  case "$SUITE" in
    safety_static_obstacles|safety_state_preservation) ;;
    *) echo "CATPROBE_ABORT: $SUITE has no measured crop box; aegis arm refused"; exit 1 ;;
  esac
  unset SE_VLA_SHIELD_OBSERVE_ONLY || true          # shield steers
  export SE_VLA_SHIELD_IMPL=vlsa_aegis SE_VLA_VLSA_DOF=3
  : "${SE_VLA_GLM_BASE_URL:=http://HOST_E:8502/v1}"
  export SE_VLA_GLM_BASE_URL
  unset ZHIPUAI_API_KEY || true
  if ! curl -sf --max-time 20 "${SE_VLA_GLM_BASE_URL%/v1}/v1/models" | grep -q glm-4.5v; then
    echo "CATPROBE_ABORT: local GLM not serving glm-4.5v at $SE_VLA_GLM_BASE_URL"; exit 1
  fi
else
  export SE_VLA_SHIELD_OBSERVE_ONLY=1               # shield computes, never steers
  unset SE_VLA_SHIELD_IMPL || true
fi

export SE_VLA_HAZARD_LOOKAHEAD_S=0.0
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="catprobe_${CELL}"

if [ "$ARM" = "aegis" ]; then
  [ -n "${SE_VLA_ARENA_SUITE:-}" ] || { echo "MISSING_REQUIRED_ENV SE_VLA_ARENA_SUITE"; exit 1; }
fi
for v in SE_VLA_SEED SE_VLA_RESULT_PATH SE_VLA_SRD_NEAR_DISTANCE_M \
         SE_VLA_SRD_RELEASE_DISTANCE_M SE_VLA_SRD_MIN_CLOSING_M_PER_STEP; do
  eval "test -n \"\${$v:-}\"" || { echo "MISSING_REQUIRED_ENV $v"; exit 1; }
done

echo "=== CatProbe $CELL suite=$SUITE level=2 tid=$TID off=$OFF arm=$ARM shield=${SE_VLA_SHIELD_IMPL:-observe_only} on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite "$SUITE" \
  --task-level 2 --task-id "$TID" --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((40000 + 20 * (SGE_TASK_ID - 1))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "CATPROBE_EXIT=$rc cell=$CELL"
[ -s "$OUT/result.json" ] || { echo "CATPROBE_CELL_FAILED $CELL"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/result.json" "$CELL" "$ARM" "$SUITE" "$TID" <<'PYCHK'
import json, sys
r = json.load(open(sys.argv[1])); md = r.get("metric_decomposition") or {}
# the suite/level/task actually executed must match what we asked for
assert str(r.get("task_suite_name")) == sys.argv[4], "suite mismatch: %s" % r.get("task_suite_name")
assert int(r.get("task_level")) == 2, "level mismatch: %s" % r.get("task_level")
assert int(r.get("task_id")) == int(sys.argv[5]), "task mismatch: %s" % r.get("task_id")
print("CATPROBE cell=%s arm=%s sr=%s official_cc=%s policy_cc=%s" % (
    sys.argv[2], sys.argv[3], r.get("successes"),
    md.get("official_cc"), md.get("policy_induced_cc")))
PYCHK
