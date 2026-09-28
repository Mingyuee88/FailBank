#!/usr/bin/env bash
#$ -N Pi0Smoke
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-2
#$ -tc 2
#$ -j y
#
# First run of the Arena-official Pi0 finetune through our runner. Two cells only: this is a
# does-it-run check, not a measurement.
#
# What could break, in order of likelihood:
#   - action_horizon is 50 for Pi0 vs 10 for Pi0.5, so every inference returns a 50-step chunk.
#     With replan_steps=1 only the first action is executed and 49 are discarded, but
#     adapt_first_action / pi05_action_adapter may assert on the chunk length.
#   - The yaml + resolution pair is new (openpi_pi0.yaml, config_resolution_pi0.json).
#   - assert_architecture now dispatches on family; it should accept pi0 and reject a pi05
#     checkpoint pointed at this config.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
OUTROOT=${WORK_ROOT}/lora_dagger/pi0_smoke

OFFS=(0 1)
i=$((SGE_TASK_ID - 1))
OFF=${OFFS[$i]}
OUT=$OUTROOT/off$OFF
mkdir -p "$OUT"
hostname > "$OUT/host.txt"

# Point the runner at the Pi0 config pair. Unset, these default to the Pi0.5 files.
export SE_VLA_OPENPI_YAML=${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/configs/evaluation/openpi_pi0.yaml
export SE_VLA_CONFIG_RESOLUTION=${SE_VLA_ROOT}/roundG/pi05_smoke/config_resolution_pi0.json
unset SE_VLA_POLICY_CHECKPOINT_DIR || true   # take the checkpoint from the pi0 yaml

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SHIELD_OBSERVE_ONLY=1
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="pi0smoke_off${OFF}"

for v in SE_VLA_SEED SE_VLA_RESULT_PATH SE_VLA_SRD_NEAR_DISTANCE_M \
         SE_VLA_SRD_RELEASE_DISTANCE_M SE_VLA_SRD_MIN_CLOSING_M_PER_STEP; do
  eval "test -n \"\${$v:-}\"" || { echo "MISSING_REQUIRED_ENV $v"; exit 1; }
done

echo "=== Pi0Smoke off=$OFF  yaml=$(basename $SE_VLA_OPENPI_YAML) on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((45000 + 20 * (SGE_TASK_ID - 1))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "PI0_SMOKE_EXIT=$rc off=$OFF"
[ -s "$OUT/result.json" ] || { echo "PI0_SMOKE_NO_RESULT off=$OFF"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/result.json" "$OFF" <<'PYCHK'
import json, sys
r = json.load(open(sys.argv[1])); md = r.get("metric_decomposition") or {}
a = r.get("architecture") or {}
print("PI0_SMOKE off=%s sr=%s cost=%s polcc=%s" % (
    sys.argv[2], r.get("successes"), r.get("cost"), md.get("policy_induced_cc")))
print("  architecture family=%s model_type=%s pi05=%s horizon=%s" % (
    a.get("architecture_family"), a.get("model_type"), a.get("pi05"), a.get("action_horizon")))
print("  chunk_length=%s inference_chunks=%s discarded=%s first_executed=%s" % (
    r.get("chunk_length"), r.get("inference_chunks"),
    r.get("discarded_actions_total"), r.get("first_actions_executed")))
assert a.get("architecture_family") == "pi0", "runner did not resolve the pi0 architecture"
print("PI0_SMOKE_VERIFIED")
PYCHK
