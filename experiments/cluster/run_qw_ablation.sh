#!/bin/bash
#$ -N QwAbl
#$ -cwd
#$ -q gpu@HOST_B
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-300
#$ -tc 4
#$ -j y
# ABLATION: does raising the TRAINING-TIME safety weight actually buy lower cost?
#
# This is the question the meeting note meant by "d weight -> CC, how much the
# self-evolving update attends to cost". It is NOT the evaluation-side `d` weighting:
# re-weighting a composite score after the fact cannot change CC, only its ranking, so
# that ablation would be circular. quiet_weight is the training-side knob and is the
# real variable.
#
# R2_q00 and R2_q20 are the same bank (se_r2, accumulated), same round (2), same 800
# steps, same base -- they differ ONLY in quiet_weight (0.0 vs 0.2). Verified from their
# metrics.json.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
L=${WORK_ROOT}/lora_dagger
OUTROOT=$L/qw_abl
EXPECT_HOST=HOST_B

# 300 = 3 arms x 50 offsets x 2 sampler-advance
i=$((SGE_TASK_ID - 1))
ai=$((i / 100)); rem=$((i % 100)); OFF=$((rem / 2)); A=$((rem % 2))
case "$ai" in
  0) ARM=base ;;
  1) ARM=q00 ;;
  2) ARM=q20 ;;
esac
CELL="${ARM}/off${OFF}_r${A}"
OUT=$OUTROOT/$CELL
mkdir -p "$OUT"
if [ -s "$OUT/result.json" ] && grep -q '"status": "pass"' "$OUT/result.json"; then echo "SKIP $CELL"; exit 0; fi
hostname > "$OUT/host.txt"
if [ "$(hostname -s)" != "$EXPECT_HOST" ]; then
  echo "HOST_MISMATCH expected=$EXPECT_HOST got=$(hostname -s) cell=$CELL"
  rm -f "$OUT/result.json"; exit 1
fi

export SE_VLA_SHIELD_OBSERVE_ONLY=1
case "$ARM" in
  base) unset SE_VLA_POLICY_CHECKPOINT_DIR || true ;;
  q00)  export SE_VLA_POLICY_CHECKPOINT_DIR=$L/nocurr/ckpt/R2_q00/offset_0 ;;
  q20)  export SE_VLA_POLICY_CHECKPOINT_DIR=$L/nocurr/ckpt/R2_q20/offset_0 ;;
esac
if [ "$ARM" != "base" ]; then
  [ -d "${SE_VLA_POLICY_CHECKPOINT_DIR}/params" ] || { echo "QW_ABORT: no $ARM checkpoint"; exit 1; }
fi

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_HAZARD_LOOKAHEAD_S=0.0
export SE_VLA_SAMPLER_ADVANCE=$A
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="qwabl_${ARM}_${OFF}_${A}"

echo "=== QwAbl $CELL arm=$ARM on $(hostname -s) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 3 --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((26000 + 3 * (SGE_TASK_ID % 1200))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "QW_EXIT=$rc cell=$CELL"
[ -s "$OUT/result.json" ] || { echo "QW_CELL_FAILED $CELL"; exit 1; }
