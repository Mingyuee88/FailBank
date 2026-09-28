#!/bin/bash
#$ -N MotivL0
#$ -cwd
#$ -q gpu@HOST_D
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-150
#$ -tc 4
#$ -j y
# MOTIVATION FIGURE: difficulty defined by TASK CATEGORY, not by level.
#
# Arena's suites give a clean 2x2 that level cannot: {obstacles, distractors} x
# {static, dynamic}. Level mixes object identity, placement and task changes together,
# so a level axis cannot say *why* something got harder. static->dynamic is one
# interpretable variable, and it is the axis on which the shield is expected to fail.
#
# ours was distilled on safety_static_obstacles only, so the distractor suites are
# zero-shot cross-category transfer -- the same cells also serve RQ4.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
L=${WORK_ROOT}/lora_dagger
OUTROOT=$L/motiv_l0
EXPECT_HOST=HOST_D

# 600 = 4 suites x 3 arms x 5 tasks x 10 offsets
# L0 is the zero-hazard anchor of the difficulty axis: same tasks, no obstacle yet.
# Only safety_static_obstacles, because that is the one suite AEGIS has a measured crop
# box for -- the anchor is only useful if all three arms can stand on it.
i=$((SGE_TASK_ID - 1))
SU=0; AI=$((i / 50)); rem2=$((i % 50))
TID=$((rem2 / 10)); OFF=$((rem2 % 10))
case "$SU" in
  0) SUITE=safety_static_obstacles;      TAG=obs_static ;;
  1) SUITE=safety_dynamic_obstacles;     TAG=obs_dynamic ;;
  2) SUITE=distractor_static_distractors;  TAG=dis_static ;;
  3) SUITE=distractor_dynamic_distractors; TAG=dis_dynamic ;;
esac
case "$AI" in
  0) ARM=base ;;
  1) ARM=aegis ;;
  2) ARM=ours ;;
esac
CELL="${TAG}/${ARM}/t${TID}_off${OFF}"
OUT=$OUTROOT/$CELL
mkdir -p "$OUT"
if [ -s "$OUT/result.json" ] && grep -q '"status": "pass"' "$OUT/result.json"; then echo "SKIP $CELL"; exit 0; fi
hostname > "$OUT/host.txt"
if [ "$(hostname -s)" != "$EXPECT_HOST" ]; then
  echo "HOST_MISMATCH expected=$EXPECT_HOST got=$(hostname -s) cell=$CELL"
  rm -f "$OUT/result.json"; exit 1
fi

case "$ARM" in
  base)  export SE_VLA_SHIELD_OBSERVE_ONLY=1; unset SE_VLA_POLICY_CHECKPOINT_DIR || true ;;
  ours)  export SE_VLA_SHIELD_OBSERVE_ONLY=1
         export SE_VLA_POLICY_CHECKPOINT_DIR=$L/nocurr/ckpt/R2_q00/offset_0 ;;
  aegis) : "${SE_VLA_GLM_BASE_URL:=http://HOST_E:8502/v1}"; export SE_VLA_GLM_BASE_URL
         export SE_VLA_SHIELD_IMPL=vlsa_aegis SE_VLA_VLSA_DOF=3
         unset ZHIPUAI_API_KEY SE_VLA_POLICY_CHECKPOINT_DIR SE_VLA_SHIELD_OBSERVE_ONLY || true ;;
esac
if [ "$ARM" = "ours" ]; then
  [ -d "${SE_VLA_POLICY_CHECKPOINT_DIR}/params" ] || { echo "MV_ABORT: no ours checkpoint"; exit 1; }
fi
# AEGIS needs a hand-measured crop box per workspace. vlsa_port.py:393 maps only
# safety_static_obstacles (hazard_avoidance is deliberately excluded -- its flat stove sits
# below the table box). On the other three suites it cannot run at all, and that limitation
# is a REPORTED RESULT, not something to paper over: the shield does not transfer across
# workspaces without re-calibration. Skip those cells instead of burning queue on them.
if [ "$ARM" = "aegis" ] && [ "$SUITE" != "safety_static_obstacles" ]; then
  echo "MV_SKIP_UNCALIBRATED arm=aegis suite=$SUITE cell=$CELL"
  printf '%s\n' "{\"status\": \"skipped\", \"reason\": \"aegis has no measured crop box for $SUITE\", \"task_suite_name\": \"$SUITE\", \"arm\": \"aegis\"}" > "$OUT/result.json"
  exit 0
fi

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_HAZARD_LOOKAHEAD_S=0.0
# run.py:1218 reads the shield suite from SE_VLA_ARENA_SUITE and defaults it to "",
# which makes _crop_suite_for raise. The shield is a process-global singleton built at
# t==0, so a cell only hits this when it is the first in its process -- which is why the
# same task both passed and failed.
export SE_VLA_ARENA_SUITE="$SUITE"
export SE_VLA_SAMPLER_ADVANCE=0
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="motiv_${TAG}_${ARM}_${TID}_${OFF}"

echo "=== Motiv $CELL suite=$SUITE arm=$ARM on $(hostname -s) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite "$SUITE" \
  --task-level 0 --task-id "$TID" --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((48000 + 3 * (SGE_TASK_ID % 1200))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "MV_EXIT=$rc cell=$CELL"
[ -s "$OUT/result.json" ] || { echo "MV_CELL_FAILED $CELL"; exit 1; }
