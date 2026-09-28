#!/usr/bin/env bash
#$ -N v11_merged
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_B
#$ -pe smp 2
#$ -t 1-150
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v11_merged/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v11_merged/batch/j.$JOB_ID.$TASK_ID.out
#
# v11_merged -- pinned base census of the merged arena, the prerequisite for any
# powered result.
#
# Power is the binding constraint on everything measured so far. L1t2 has 11 base
# failures on this host and only 6 of them are held out from the harvest, so a fix
# rate there carries a 95% Wilson interval of roughly [3%, 56%] and cannot separate
# any two arms. Reaching 80% power at the observed effect size needs on the order
# of 39 base-failure offsets.
#
# The merged arena L1t1 + L1t3 + L1t4 (lemon / onion / tomato, all level-1
# place-on-bowl) was chosen from the v8_map census:
#
#     arena   SR     failures  mean polCC   polCC on failures
#     L1t1    0.780     11        34.8          149.7
#     L1t3    0.740     13        37.6          135.6
#     L1t4    0.820      9        48.5          220.4
#     ----------------------------------------------------
#     merged            33       ~40
#
# Excluded and why: L0 and L1t0 have polCC identically 0 on their failures, so
# there is no attributable cost to remove; L2t0 (SR 0.060) and L2t1 (SR 0.000) are
# zero-competence traps where "did nothing" and "tried and failed" are
# indistinguishable; L2t2 has only 4 failures.
#
# WHY RE-SCAN. v8_map ran under `-l h=(HOST_D*|HOST_E*)` and earlier under
# a mixed set, and swapping GPU model alone flips SR on about 1 offset in 12
# (L1t2 off35 succeeds on A6000 and fails on L40S). So the v8_map failure lists
# are host-specific and cannot define the offset partition for experiments that
# will run on HOST_B. This job produces the authoritative list on the host
# where the follow-up will actually run.
#
# Also gives the paired base reference for every future arm on this arena, and
# doubles as the cross-task generalisation set: three different manipulated
# objects, one memory.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

TASKS=(1 3 4)     # lemon / onion / tomato, all level 1
NOFF=50
i=$((SGE_TASK_ID - 1))
TID=${TASKS[$((i / NOFF))]}
OFF=$((i % NOFF))

OUT=$LR/v11_merged/L1t${TID}/off${OFF}
mkdir -p "$OUT" $LR/v11_merged/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="merged_base_L1t${TID}_off${OFF}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0

echo "=== v11 base L1t${TID} off${OFF} on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id "$TID" --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((4000 + 25 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
echo "V11_EXIT=$? L1t$TID off=$OFF"
