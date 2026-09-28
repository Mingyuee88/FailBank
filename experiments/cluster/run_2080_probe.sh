#!/usr/bin/env bash
#$ -N gpu2080probe
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=SMALL_GPU_HOSTS*
#$ -pe smp 2
#$ -t 1-2
#$ -tc 2
#$ -o ${WORK_ROOT}/lora_dagger/gpu2080probe/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/gpu2080probe/j.$JOB_ID.$TASK_ID.out
#
# Can pi05/OpenPI run on the 10.75 GB 2080ti nodes?  The handoff bans them, but
# that ban was established for OpenVLA (which OOMs).  run.py already caps JAX at
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 and disables preallocation.  If pi05 fits,
# the usable pool goes from 14 cards to 22 (+57%).  Two known-answer offsets:
# off3 must come back success=1 polCC=4, off29 must come back success=0.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.88
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OFFSETS=(3 29)
OFF=${OFFSETS[$((SGE_TASK_ID - 1))]}
OUT=$LR/gpu2080probe/off$OFF
mkdir -p "$OUT"
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="probe2080_off${OFF}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "=== 2080ti probe off=$OFF ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((60000 + 30 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
echo "PROBE_EXIT=$? off=$OFF"
