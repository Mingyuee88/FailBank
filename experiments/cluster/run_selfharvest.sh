#!/usr/bin/env bash
#$ -N selfharv
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-20
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/selfharvest/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/selfharvest/batch/j.$JOB_ID.$TASK_ID.out
# Phase A: harvest POLICY SELF-EXPERIENCE with the shield OFF.
#
# Why this and not shield overrides: on this task the shield is net harmful
# (matched n=20: SR 0.75 -> 0.50, policy cost 57 -> 119.5; 3 genuine rescues
# against 8 catastrophic inductions), so its corrections are a bad teacher.
# The writer's other path is exactly what we want: with aegis disabled,
# _writer.observe(..., allow_recovery=not aegis_enabled) turns ON self-recovery
# capture - the gate flags a collision course, and if the POLICY ITSELF then
# increases separation without adding cost, that policy action is recorded.
# Provenance srd_self_recovery_pending_probe. Every previous run had the shield
# on, which disabled this path entirely.
#
# Offsets 10-29 only. 0-9 stay held out for evaluation.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers PI05_SERVER_TIMEOUT_SEC=600
cd ${SE_VLA_ROOT}

OFF=$((9 + SGE_TASK_ID))     # 10..29
SEED=17
ROOT=${WORK_ROOT}/lora_dagger/selfharvest
SEED_CK=${WORK_ROOT}/lora_dagger/round0_control/seed_checkpoint.json
OUT=$ROOT/off${OFF}_s${SEED}
mkdir -p "$OUT" "$ROOT/batch"

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED="$SEED" SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="selfharv_off${OFF}_s${SEED}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_SRD_TTC_MIN_STEPS=1 SE_VLA_SRD_TTC_MAX_STEPS=8
export SE_VLA_SRD_COOLDOWN_STEPS=8 SE_VLA_SRD_DUTY_CAP=0.15
export SE_VLA_SRD_RECOVERY_HORIZON_STEPS=12 SE_VLA_SRD_MIN_RECOVERY_SEPARATION_M=0.01
export SE_VLA_RELATION_MEMORY_FEATURE_KEYS="cost_pair_min_distance,cost_pair_delta_xy[0],cost_pair_delta_xy[1],fall_max_angle_abs,episode_step_index"
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned

# writer ON, shield OFF -> self-recovery capture becomes reachable
export SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=1
export SE_VLA_AEGIS_SOURCE=0
export SE_VLA_ADAPTER_SAVE_CHECKPOINT_PATH="$OUT/harvest_checkpoint.json"

echo "=== self-harvest (shield OFF) off=$OFF seed=$SEED ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 2 --task-id 4 --offset "$OFF" --seed "$SEED" \
  --replan-steps 1 --trials 1 --arm pruned_memory --checkpoint "$SEED_CK" \
  --port-base $((42000 + SGE_TASK_ID)) --array-task-id "$SGE_TASK_ID" --server-attempts 3
echo "SELFHARV_EXIT=$? off=$OFF"
