#!/usr/bin/env bash
#$ -N v7_confirm
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-27
#$ -tc 8
#$ -o ${WORK_ROOT}/lora_dagger/v7_confirm/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v7_confirm/batch/j.$JOB_ID.$TASK_ID.out
# THE MISSING LINK: evaluate base + memory with the shield OFF.
#
# stage2_arms_param.py never does this. Its probe runs `--arm base` with no
# --checkpoint, because that probe exists to supply the counterfactual "what would
# have happened without the shield" that the credit rule needs. So the harvest and
# credit machinery has been exercised for months while the deployment arm -- the
# only thing that measures the METHOD -- was never run.
#
# Two axes, both preregistered before this job was submitted:
#   CONFIRMATION ONLY. The winning configuration (new key, no gate, h5) reached
#   whole-task SR 0.796 vs base 0.745 -- but it was the best of 8 configurations,
#   selected using the same 10 probe offsets. These 27 base-success offsets were
#   never probed and never used for that selection, so retention measured here is
#   free of the selection. Fix rate cannot be confirmed the same way: all 10
#   base-failure offsets are already spent (5 adapted, 5 probed).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
ARMS=(newkey-h5)
OFFSETS=(3 7 8 12 13 14 15 16 18 19 21 22 23 25 28 30 32 33 34 35 37 39 40 41 45 46 47)
i=$((SGE_TASK_ID-1))
ARM=${ARMS[0]}
OFF=${OFFSETS[$i]}
CK=$(ls -d $LR/v7_l1t2/$ARM/round_*/settled_checkpoint.json 2>/dev/null | tail -1)
[ -n "$CK" ] || { echo "no settled checkpoint for $ARM"; exit 1; }
OUT=$LR/v7_confirm/$ARM/off$OFF
mkdir -p "$OUT" $LR/v7_confirm/batch
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="deploy_${ARM}_off${OFF}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_FEATURE_KEYS="cost_pair_min_distance,cost_pair_delta_xy[0],cost_pair_delta_xy[1],eef_hazard_distance,eef_target_distance,episode_step_index"
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
# deployment: shield OFF, memory read-only, no new writes, no credit
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
# 2x2 ablation: retrieval key (0.792 -> 0.875 AUC) x intervene/abstain gate.
# The key decides WHETHER a state is retrieved at all; the gate is a blunt filter
# applied after retrieval. The previous round showed the two are same-direction
# levers -- suppressing firing removed the fixes along with the damage -- so the
# key has to be tested both with and without the gate.
# gate deliberately OFF in this arm

echo "=== deploy arm=$ARM offset=$OFF ckpt=$CK ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm pruned_memory --checkpoint "$CK" \
  --port-base $((64000 + 30 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
echo "DEPLOY_EXIT=$? arm=$ARM off=$OFF"
