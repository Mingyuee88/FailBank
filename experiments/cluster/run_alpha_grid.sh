#!/usr/bin/env bash
#$ -N aeg_grid
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-40
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/aegis_grid/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/aegis_grid/batch/j.$JOB_ID.$TASK_ID.out
# Clean alpha sweep at FIXED geometry, on L2_t4.
#
# The previous run varied alpha AND the radii together, so its alpha=10 numbers
# are confounded: clearance went 0.03+0.04+0.02=0.09 -> 0.06+0.08+0.0=0.14, which
# shrinks the barrier h = d - clearance and makes the constraint bind MORE, in the
# opposite direction to the alpha effect. Here geometry is held fixed within each
# half of the grid so alpha is the only thing moving.
#
# The point is to find AEGIS's BEST achievable configuration and use that as the
# baseline, so the comparison cannot be dismissed as a mis-tuned incumbent.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers PI05_SERVER_TIMEOUT_SEC=600
cd ${SE_VLA_ROOT}

ALPHAS=(1 3 10 30)
i=$((SGE_TASK_ID-1))
GEO=$((i/20))            # 0 = tight (as originally run), 1 = code default
rem=$((i%20))
ALPHA=${ALPHAS[$((rem/5))]}
OFF=$((rem%5))
if [ "$GEO" -eq 0 ]; then EEF=0.03; OBS=0.04; MAR=0.02; GNAME=tight
else EEF=0.06; OBS=0.08; MAR=0.0; GNAME=default; fi

SUITE=safety_static_obstacles; LEVEL=2; TASK=4; SEED=23
ROOT=${WORK_ROOT}/lora_dagger/aegis_grid
BASE_CKPT=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
OUT=$ROOT/${GNAME}_a${ALPHA}/off${OFF}
mkdir -p "$OUT" "$ROOT/batch"

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED="$SEED" SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_POLICY_CHECKPOINT_DIR="$BASE_CKPT"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1
export SE_VLA_AEGIS_ALPHA="$ALPHA"
export SE_VLA_AEGIS_EEF_RADIUS="$EEF" SE_VLA_AEGIS_ORACLE_RADIUS="$OBS" SE_VLA_AEGIS_MARGIN="$MAR"

echo "=== grid geo=$GNAME alpha=$ALPHA off=$OFF ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite "$SUITE" \
  --task-level "$LEVEL" --task-id "$TASK" --offset "$OFF" --seed "$SEED" \
  --replan-steps 1 --trials 1 --arm base --port-base $((38000 + SGE_TASK_ID)) \
  --array-task-id "$SGE_TASK_ID" --server-attempts 3
echo "GRID_EXIT=$? geo=$GNAME alpha=$ALPHA off=$OFF"
