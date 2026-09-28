#!/usr/bin/env bash
#$ -N armchk
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-12
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/armcheck/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/armcheck/batch/j.$JOB_ID.$TASK_ID.out
# Integrity check before any matched-pair analysis.
#
# Same offset, same seed, shield OFF in both cases:
#   --arm base                              -> off10 gave SR=0, polC=227
#   --arm pruned_memory with EMPTY memory    -> off10 gave SR=1, polC=8
# An empty memory should be behaviourally identical to base. If it is not, every
# matched comparison we plan to build on these runs is invalid.
#
# 3 offsets x 2 arms x 2 repetitions. Repetitions use an identical seed, so any
# spread within a cell is pure run-to-run nondeterminism, and any consistent gap
# between cells is a genuine arm effect.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers PI05_SERVER_TIMEOUT_SEC=600
cd ${SE_VLA_ROOT}

OFFSETS=(10 11 14)
i=$((SGE_TASK_ID-1))
OFF=${OFFSETS[$((i/4))]}
rem=$((i%4))
ARMIDX=$((rem/2))
REP=$((rem%2))
if [ "$ARMIDX" -eq 0 ]; then ARM=base; ANAME=base; else ARM=pruned_memory; ANAME=prunedmem_empty; fi

SEED=17
ROOT=${WORK_ROOT}/lora_dagger/armcheck
SEED_CK=${WORK_ROOT}/lora_dagger/round0_control/seed_checkpoint.json
OUT=$ROOT/off${OFF}_${ANAME}_rep${REP}
mkdir -p "$OUT" "$ROOT/batch"

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED="$SEED" SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="armchk_off${OFF}_${ANAME}_r${REP}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_FEATURE_KEYS="cost_pair_min_distance,cost_pair_delta_xy[0],cost_pair_delta_xy[1],fall_max_angle_abs,episode_step_index"
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
# shield off in every cell; SRD telemetry on so both arms log identically
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=1

CMD=(roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles
     --task-level 2 --task-id 4 --offset "$OFF" --seed "$SEED"
     --replan-steps 1 --trials 1 --arm "$ARM"
     --port-base $((43000 + SGE_TASK_ID)) --array-task-id "$SGE_TASK_ID" --server-attempts 3)
if [ "$ARM" = "pruned_memory" ]; then CMD+=(--checkpoint "$SEED_CK"); fi

echo "=== armcheck off=$OFF arm=$ANAME rep=$REP ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python "${CMD[@]}"
echo "ARMCHK_EXIT=$? off=$OFF arm=$ANAME rep=$REP"
