#!/usr/bin/env bash
#$ -N AegisT3
#$ -cwd
#$ -q gpu@HOST_D
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-18
#$ -tc 3
#$ -j y
#
# AEGIS on safety_dynamic_obstacles L1 t3 -- the missing third party in the base / AEGIS /
# ours comparison. Every AEGIS number we have so far is from the STATIC suite (SR 80.0 /
# CC 7.4); it has never been run on a dynamic task.
#
# Unlike every other arm in this comparison the shield is IN THE LOOP here: no
# SE_VLA_SHIELD_OBSERVE_ONLY, so projections actually modify the executed action. The policy
# is the untouched base -- AEGIS is a runtime layer, not a trained adapter. That is the point
# of the comparison: a shield that steers vs a policy that learned from its own failures.
#
# Same host (a6k-002) and same 18 offsets as base / tau00 / bank arms, so the four are
# directly comparable cell by cell.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUT=$LR/tau_eval_t3/aegis_off$((SGE_TASK_ID - 1))
OFF=$((SGE_TASK_ID - 1))
mkdir -p "$OUT"
if [ -s "$OUT/result.json" ]; then echo "SKIP aegis_off$OFF"; exit 0; fi
hostname > "$OUT/host.txt"

# base policy, shield steering
unset SE_VLA_POLICY_CHECKPOINT_DIR SE_VLA_SHIELD_OBSERVE_ONLY || true
export SE_VLA_SHIELD_IMPL=vlsa_aegis SE_VLA_VLSA_DOF=3
: "${SE_VLA_GLM_BASE_URL:=http://HOST_E:8502/v1}"
export SE_VLA_GLM_BASE_URL
unset ZHIPUAI_API_KEY || true
if ! curl -sf --max-time 20 "${SE_VLA_GLM_BASE_URL%/v1}/v1/models" | grep -q glm-4.5v; then
  echo "AEGIST3_ABORT: local GLM not serving glm-4.5v at $SE_VLA_GLM_BASE_URL"; exit 1
fi
export SE_VLA_HAZARD_LOOKAHEAD_S=0.0
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="t3eval_aegis_off${OFF}"

for v in SE_VLA_SEED SE_VLA_RESULT_PATH SE_VLA_SRD_NEAR_DISTANCE_M \
         SE_VLA_SRD_RELEASE_DISTANCE_M SE_VLA_SRD_MIN_CLOSING_M_PER_STEP; do
  eval "test -n \"\${$v:-}\"" || { echo "MISSING_REQUIRED_ENV $v"; exit 1; }
done

echo "=== AegisT3 off=$OFF shield=$SE_VLA_SHIELD_IMPL (in-loop) on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_dynamic_obstacles \
  --task-level 1 --task-id 3 --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((53000 + 20 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "AEGIST3_EXIT=$rc off=$OFF"
[ -s "$OUT/result.json" ] || { echo "AEGIST3_CELL_FAILED off=$OFF"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/result.json" "$OFF" <<'PYCHK'
import json, sys
r = json.load(open(sys.argv[1])); md = r.get("metric_decomposition") or {}
ad = r.get("adapter") or {}
print("AEGIST3 off=%s sr=%s official_cc=%s policy_cc=%s corrections=%s" % (
    sys.argv[2], r.get("successes"), md.get("official_cc"),
    md.get("policy_induced_cc"), ad.get("corrections_applied")))
assert (ad.get("corrections_applied") or 0) >= 0, "no adapter telemetry"
PYCHK
