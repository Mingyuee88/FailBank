#!/bin/bash
#$ -N DynR3
#$ -cwd
#$ -q gpu@HOST_B
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-200
#$ -tc 4
#$ -j y
# Round 1 vs round 3 on safety_dynamic_obstacles L2, task 0.
#
# WHY THIS RUN EXISTS. The earlier 100-cell dynamic column came out null (4/2, p=0.69) and
# I read that as "the suite has no measurable band". That reading was wrong on two counts:
#   1. it used se_dyn_L2t0/DYN_L2t0 -- the ROUND-1 checkpoint, 5024 rows -- while the static
#      column's ours arm is round 2. Unequal rounds.
#   2. it spread that checkpoint over all five L2 tasks, but the multi-round dynamic assets
#      (dcol_L2t0 -> _r2 -> _r3) exist only for tid=0.
# tid=0 is also the one L2 dynamic task with headroom (base SR 70.0, 3/10 improvable).
#
# Arms: base / round1 (5024 rows) / round3 (bank_dyn_r3b, 24803 rows, flow 0.980, accepted).
# AEGIS is absent by construction: no measured crop box for this suite (vlsa_port.py:393).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
L=${WORK_ROOT}/lora_dagger
OUTROOT=$L/dyn_r3_t0
EXPECT_HOST=HOST_B
SUITE=safety_dynamic_obstacles

# 150 = 3 arms x 50 offsets, task fixed at tid 0
i=$((SGE_TASK_ID - 1))
AI=$((i / 50)); OFF=$((i % 50)); TID=0
case "$AI" in
  0) ARM=base ;;
  1) ARM=r1 ;;
  2) ARM=r3 ;;
  3) ARM=r2 ;;
  *) echo "no arm"; exit 0 ;;
esac
CELL="${ARM}/off${OFF}"
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
  r1)   export SE_VLA_POLICY_CHECKPOINT_DIR=$L/se_dyn_L2t0/ckpt/DYN_L2t0/offset_0 ;;
  r3)   export SE_VLA_POLICY_CHECKPOINT_DIR=$L/se_bank_r3b_ho/ckpt/BANK3_700/offset_0 ;;
  # round 2: bank_dyn_r2, 15303 rows. BANK700 (700 steps, qw=0) matches BANK3_700's
  # recipe; the BANK800Q01/Q02 variants would change quiet_weight as well as round.
  r2)   export SE_VLA_POLICY_CHECKPOINT_DIR=$L/se_bank_r2_ho/ckpt/BANK700/offset_0 ;;
esac
if [ "$ARM" != "base" ]; then
  [ -d "${SE_VLA_POLICY_CHECKPOINT_DIR}/params" ] || { echo "DR3_ABORT: no params for $ARM"; exit 1; }
fi

export SE_VLA_ARENA_SUITE="$SUITE"
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_HAZARD_LOOKAHEAD_S=0.0
export SE_VLA_SAMPLER_ADVANCE=0
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="dynr3_${ARM}_${OFF}"

echo "=== DynR3 $CELL arm=$ARM suite=$SUITE on $(hostname -s) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite "$SUITE" \
  --task-level 2 --task-id "$TID" --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((54000 + 3 * (SGE_TASK_ID % 1200))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "DR3_EXIT=$rc cell=$CELL"
[ -s "$OUT/result.json" ] || { echo "DR3_CELL_FAILED $CELL"; exit 1; }
