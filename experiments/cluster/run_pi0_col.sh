#!/usr/bin/env bash
#$ -N Pi0Col
#$ -cwd
#$ -q gpu@HOST_E
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-32
#$ -tc 4
#$ -j y
#
# Round-1 collection for the Pi0 arm, on safety_static_obstacles level_1 task 2 only.
#
# Task choice is forced by the baseline: across 150 cells Pi0 produced policy-induced cost on
# 28/50 cells of t2 and on 0/50 of both t1 and t4. Pi0 is a conservative policy -- SR 72.0 with
# CC 0.8 and 27.3% safe-fail, against Pi0.5's SR 81.7 / CC 46.4 / 0.7% safe-fail -- so t2 is
# the only task in this trio where a failure bank can be collected at all.
#
# Offsets 0-31 only, leaving 32-49 for a clean holdout, the same split the Pi0.5 holdout used.
# PINNED to a6k-003: every arm of this comparison must sit on one GPU model.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
OUTROOT=${WORK_ROOT}/lora_dagger/pi0_col_t2

TIDS=(2)
i=$((SGE_TASK_ID - 1))
t=0; OFF=$i
TID=${TIDS[$t]}
CELL="t${TID}_off${OFF}"
OUT=$OUTROOT/$CELL
mkdir -p "$OUT"
if [ -s "$OUT/result.json" ]; then echo "SKIP $CELL"; exit 0; fi
hostname > "$OUT/host.txt"

export SE_VLA_OPENPI_YAML=${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/configs/evaluation/openpi_pi0.yaml
export SE_VLA_CONFIG_RESOLUTION=${SE_VLA_ROOT}/roundG/pi05_smoke/config_resolution_pi0.json
unset SE_VLA_POLICY_CHECKPOINT_DIR || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SHIELD_OBSERVE_ONLY=1
export SE_VLA_PHASE1_RECORD=1 SE_VLA_PHASE1_CANDIDATES=1
export SE_VLA_PHASE1_ROOT=$OUTROOT/records
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="pi0col_${CELL}"

for v in SE_VLA_SEED SE_VLA_RESULT_PATH SE_VLA_SRD_NEAR_DISTANCE_M \
         SE_VLA_SRD_RELEASE_DISTANCE_M SE_VLA_SRD_MIN_CLOSING_M_PER_STEP; do
  eval "test -n \"\${$v:-}\"" || { echo "MISSING_REQUIRED_ENV $v"; exit 1; }
done

echo "=== Pi0Col $CELL on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id "$TID" --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((48000 + 20 * (SGE_TASK_ID - 1))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "PI0COL_EXIT=$rc cell=$CELL"
[ -s "$OUT/result.json" ] || { echo "PI0COL_CELL_FAILED $CELL"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/result.json" "$CELL" <<'PYCHK'
import json, sys
r = json.load(open(sys.argv[1])); md = r.get("metric_decomposition") or {}
a = r.get("architecture") or {}
assert a.get("architecture_family") == "pi0", "not the pi0 architecture"
assert a.get("signature_keys_present") is True
print("PI0COL cell=%s sr=%s offcc=%s polcc=%s" % (
    sys.argv[2], r.get("successes"), md.get("official_cc"), md.get("policy_induced_cc")))
PYCHK
