#!/usr/bin/env bash
#$ -N DynScan
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-64
#$ -tc 4
#$ -j y
#
# DISCRIMINABILITY SCAN over safety_dynamic_obstacles.
#
# Reason this exists: level_1 task 0 measured base SR 95.0% (19/20) with policy-induced cost on
# 1/20 cells. That is a ceiling, not a benchmark -- a self-evolution pipeline that learns from
# collected failures has nothing to collect. Before spending a full pipeline on this suite we
# measure, per task, whether the base policy actually fails and actually incurs POLICY-induced
# cost (official_cc includes reset contamination: teapots/mugs that topple at reset).
#
# Covers level_1 tasks 1-4 and level_2 tasks 0-3, 8 offsets each. A task is usable only if base
# SR leaves headroom AND policy_induced_cc is nonzero on a meaningful fraction of cells.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/dynscan

# 8 tasks x 8 offsets, row-major
LEVELS=(1 1 1 1 2 2 2 2)
TIDS=(1 2 3 4 0 1 2 3)
OFFS=(0 1 2 3 4 5 6 7)

i=$((SGE_TASK_ID - 1))
t=$((i / 8)); o=$((i % 8))
[ "$t" -lt "${#TIDS[@]}" ] || { echo "no cell"; exit 0; }
LV=${LEVELS[$t]}; TID=${TIDS[$t]}; OFF=${OFFS[$o]}
CELL="L${LV}t${TID}_off${OFF}"
OUT=$OUTROOT/$CELL
mkdir -p "$OUT"
if [ -s "$OUT/result.json" ]; then echo "SKIP $CELL"; exit 0; fi
hostname > "$OUT/host.txt"
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
unset SE_VLA_POLICY_CHECKPOINT_DIR || true   # base policy: the headroom we measure is the base's
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SHIELD_OBSERVE_ONLY=1          # observe only: we are measuring the base, not steering it
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="dynscan_${CELL}"
# run.py reads these with os.environ[...] and no default; a missing one dies after the whole
# simulator stack has loaded, which is how the first submission burned 64 cells on a KeyError.
for v in SE_VLA_SEED SE_VLA_RESULT_PATH SE_VLA_SRD_NEAR_DISTANCE_M \
         SE_VLA_SRD_RELEASE_DISTANCE_M SE_VLA_SRD_MIN_CLOSING_M_PER_STEP; do
  eval "test -n \"\${$v:-}\"" || { echo "MISSING_REQUIRED_ENV $v"; exit 1; }
done
echo "=== DynScan $CELL (level=$LV tid=$TID off=$OFF) base policy on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_dynamic_obstacles \
  --task-level "$LV" --task-id "$TID" --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((44000 + 20 * (SGE_TASK_ID - 1))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "SCAN_EXIT=$rc cell=$CELL"
[ -s "$OUT/result.json" ] || { echo "SCAN_CELL_FAILED $CELL"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/result.json" "$CELL" <<'PYCHK'
import json, sys
res = json.load(open(sys.argv[1])); md = res.get("metric_decomposition") or {}
print("SCAN_RESULT cell=%s sr=%s off_cc=%s pol_cc=%s cost=%s" % (
    sys.argv[2], res.get("successes"), md.get("official_cc"),
    md.get("policy_induced_cc"), res.get("cost")))
PYCHK
