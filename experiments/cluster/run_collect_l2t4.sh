#!/usr/bin/env bash
#$ -N c_l2t4
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-40
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/phase1_L2t4/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/phase1_L2t4/batch/j.$JOB_ID.$TASK_ID.out
# Phase1 experience collection on the main battlefield: safety_static_obstacles L2 t4
# (pick_the_tomato_and_place_it_on_the_bowl_2). base there = SR 0.60 / polC 99.0,
# i.e. real headroom: cut cost without losing success.
#
# OFFSET DISCIPLINE (matches the preregistration in make_manifest.py):
#   offsets 0-9   = held-out EVAL only, never collected on (0-4 already have
#                   base+shield baselines from step1_confirm)
#   offsets 10-29 = collection only
# Aegis MUST be on: _phase1_capture fails closed when proxy.aegis_result is None.
# SRD writes stay OFF so rollout dynamics match the measured shield arm exactly;
# this round produces the offline dataset, not runtime memory.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers PI05_SERVER_TIMEOUT_SEC=600
cd ${SE_VLA_ROOT}

SUITE=safety_static_obstacles; LEVEL=2; TASK=4
SEEDS=(17 19)
i=$((SGE_TASK_ID-1))
SEED=${SEEDS[$((i/20))]}
OFF=$((10 + i%20))

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

# Aegis identical to the step1_confirm shield arm -- data distribution must match
# the arm we already characterised.
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02

# Phase1 recorder ON (full: candidates + blobs)
export SE_VLA_PHASE1_RECORD=1 SE_VLA_PHASE1_CANDIDATES=1
export SE_VLA_PHASE1_ROOT="$ROOT/records"
export SE_VLA_PHASE1_TRACE="$OUT"

echo "=== collect L2t4 off=$OFF seed=$SEED ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite "$SUITE" \
  --task-level "$LEVEL" --task-id "$TASK" --offset "$OFF" --seed "$SEED" \
  --replan-steps 1 --trials 1 --arm base --port-base $((35000 + SGE_TASK_ID)) \
  --array-task-id "$SGE_TASK_ID" --server-attempts 3
echo "COLLECT_EXIT=$? off=$OFF seed=$SEED"
