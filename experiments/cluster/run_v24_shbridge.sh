#!/usr/bin/env bash
#$ -N v24_shbridge
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 2
#$ -t 1-36
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v24_shbridge/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v24_shbridge/batch/j.$JOB_ID.$TASK_ID.out
#
# v24_shbridge -- the shield's 47-offset profile through ONE pipeline.
#
# Adversarial review found a real defect in how that profile was assembled: it
# spliced v9_power/shield (base-FAILURE offsets, run as a deliberate upper-bound
# measurement) with v16_negcollect (base-SUCCESS offsets, run as teacher
# collection). Same host, same verified recipe, but the two pipelines cover
# disjoint strata with no overlapping offset, so a batch/pipeline effect could not
# be detected even in principle. That makes the resulting "shield fixes 10/11 and
# destroys 9/36, net +1" an exploratory recombination, not a confirmatory
# comparison -- and it is the number the whole misalignment claim rests on.
#
# This job removes the splice: the SAME script that produced v9_power/shield, run
# on the 36 base-SUCCESS offsets it never covered. Combined with the existing 11
# failure cells that yields a single-pipeline 47-offset profile, and comparing it
# against v16_negcollect on the same offsets also measures the pipeline effect
# directly.
#
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

# the 36 base-SUCCESS offsets on HOST_A, read off v8_clean/base
OKS=(1 2 3 4 6 7 8 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 28 30 31 32 33 34 37 38 39 40 41 44 45 46)
MODE=shield
OFF=${OKS[$((SGE_TASK_ID - 1))]}

OUT=$LR/v24_shbridge/shield/off$OFF
mkdir -p "$OUT" $LR/v24_shbridge/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="power_${MODE}_off${OFF}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
PORT=$((42000 + 25 * SGE_TASK_ID))

if [ "$MODE" = shield ]; then
  # frozen policy + the CBF shield, no memory at all. Verified recipe from the
  # harvest manifest: alpha 3 / eef 0.03 / oracle 0.04 / margin 0.02.
  export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
  export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04
  export SE_VLA_AEGIS_MARGIN=0.02 SE_VLA_AEGIS_MAX_TRANSLATION=1.0 SE_VLA_AEGIS_OVERRIDE_EPS=1e-6
  export SE_VLA_SRD_COOLDOWN_STEPS=8 SE_VLA_SRD_DUTY_CAP=0.15
  export SE_VLA_SRD_TTC_MIN_STEPS=1 SE_VLA_SRD_TTC_MAX_STEPS=8
  export SE_VLA_SRD_RECOVERY_HORIZON_STEPS=12 SE_VLA_SRD_MIN_RECOVERY_SEPARATION_M=0.01
  export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
  echo "=== v24 SHIELD BRIDGE off=$OFF on $(hostname) ==="
  PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
    roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
    --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
    --replan-steps 1 --trials 1 --arm base \
    --port-base $PORT --array-task-id 1 --server-attempts 3
  echo "BRIDGE_EXIT=$? off=$OFF"; exit 0
fi

echo "unreachable"
