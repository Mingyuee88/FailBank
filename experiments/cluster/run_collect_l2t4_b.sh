#!/usr/bin/env bash
#$ -N c_l2t4b
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-20
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/phase1_L2t4/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/phase1_L2t4/batch/j.$JOB_ID.$TASK_ID.out
# Phase1 collection round B on safety_static_obstacles L2 t4.
#
# select_init_state_index does (base+offset) % num_initial_states, and this task
# has exactly 50 initial states -> offsets 0-49 are the entire universe.
#   0-9   = held-out EVAL, never collected on
#   10-29 = round A (done: 40 episodes, 2 seeds)
#   30-49 = round B (this script) -- the LAST available data on this task
# Round A showed 8/20 offsets produced BIT-IDENTICAL rollouts across seeds 17/19,
# so a second seed buys nothing on those. Round B spends the budget on distinct
# offsets with a single seed instead: 20 offsets x 1 seed = 20 distinct trajectories.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers PI05_SERVER_TIMEOUT_SEC=600
cd ${SE_VLA_ROOT}

SUITE=safety_static_obstacles; LEVEL=2; TASK=4
SEED=17
OFF=$((29 + SGE_TASK_ID))   # task 1..20 -> offset 30..49

if [ "$OFF" -lt 30 ] || [ "$OFF" -gt 49 ]; then
  echo "FATAL: offset $OFF outside collection range 30-49"; exit 1
fi

ROOT=${WORK_ROOT}/lora_dagger/phase1_L2t4
BASE_CKPT=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
OUT=$ROOT/runs/off${OFF}_s${SEED}
mkdir -p "$OUT" "$ROOT/batch" "$ROOT/records"

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED="$SEED" SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="L2t4_off${OFF}_s${SEED}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_POLICY_CHECKPOINT_DIR="$BASE_CKPT"

export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02

export SE_VLA_PHASE1_RECORD=1 SE_VLA_PHASE1_CANDIDATES=1
export SE_VLA_PHASE1_ROOT="$ROOT/records"
export SE_VLA_PHASE1_TRACE="$OUT"

echo "=== collect-B L2t4 off=$OFF seed=$SEED ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite "$SUITE" \
  --task-level "$LEVEL" --task-id "$TASK" --offset "$OFF" --seed "$SEED" \
  --replan-steps 1 --trials 1 --arm base --port-base $((36000 + SGE_TASK_ID)) \
  --array-task-id "$SGE_TASK_ID" --server-attempts 3
echo "COLLECT_EXIT=$? off=$OFF seed=$SEED"
