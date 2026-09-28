#!/bin/bash
#$ -N CurveEv
#$ -cwd
#$ -q gpu@HOST_B
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-600
#$ -tc 4
#$ -j y
# Round curve 1-5 evaluated on L1-t3, the one static task with headroom
# (base SR 72.0, polCC 27.18; t0 is saturated at SR 90 / polCC 0.12 and cannot move).
#
# All five arms share ONE quiet_weight (0.2) and ACCUMULATED banks, so the only variable
# across arms is round count. Pinned to a single host: GPU model alone has moved SR by 29
# points on an identical policy, which would swamp any round effect.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
L=${WORK_ROOT}/lora_dagger
OUTROOT=$L/curve_t3
EXPECT_HOST=HOST_B

i=$((SGE_TASK_ID - 1))
ai=$((i / 100)); rem=$((i % 100)); OFF=$((rem / 2)); A=$((rem % 2))
case "$ai" in
  0) ARM=base ;;
  1) ARM=r1 ;;
  2) ARM=r2 ;;
  3) ARM=r3 ;;
  4) ARM=r4 ;;
  5) ARM=r5 ;;
  *) echo "no arm"; exit 0 ;;
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
  r1) export SE_VLA_POLICY_CHECKPOINT_DIR=$L/round_curve/ckpt/R1_q20/offset_0 ;;
  r2) export SE_VLA_POLICY_CHECKPOINT_DIR=$L/nocurr/ckpt/R2_q20/offset_0 ;;
  r3) export SE_VLA_POLICY_CHECKPOINT_DIR=$L/round_curve/ckpt/R3c_q20/offset_0 ;;
  r4) export SE_VLA_POLICY_CHECKPOINT_DIR=$L/round_curve/ckpt/R4_q20/offset_0 ;;
  r5) export SE_VLA_POLICY_CHECKPOINT_DIR=$L/round_curve/ckpt/R5_q20/offset_0 ;;
esac
if [ "$ARM" != "base" ]; then
  [ -d "${SE_VLA_POLICY_CHECKPOINT_DIR}/params" ] || { echo "CE_ABORT: no params for $ARM"; exit 1; }
fi

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_HAZARD_LOOKAHEAD_S=0.0
# adv is an ENV var, not a CLI flag: run.py takes only the flags used below and would
# argparse-error on --sampler-advance. run_multitask.sh sets it exactly this way.
export SE_VLA_SAMPLER_ADVANCE=$A
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="curve_${ARM}_${OFF}_${A}"

echo "=== CurveEv $CELL arm=$ARM adv=$A on $(hostname -s) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 3 --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((36000 + 3 * (SGE_TASK_ID % 1200))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "CE_EXIT=$rc cell=$CELL"
[ -s "$OUT/result.json" ] || { echo "CE_CELL_FAILED $CELL"; exit 1; }
