#!/usr/bin/env bash
#$ -N v9_rounds
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_B
#$ -pe smp 2
#$ -t 1-60
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v9_rounds/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v9_rounds/batch/j.$JOB_ID.$TASK_ID.out
#
# v9_rounds -- THE SELF-EVOLUTION CURVE.  Preregistered criterion 3, never yet run.
#
# Everything measured so far is the FROZEN arm: deployment loads a settled
# checkpoint with SE_VLA_SRD_WRITES_ENABLED=0 and CREDIT_ENABLED=0, so the memory
# never updates while being evaluated.  Criterion 3 ("adaptive beats frozen by
# >=10%") has therefore never been tested.  The 08-10 handoff records a three-arm
# job that nominally covered it, but its probe ran `--arm base` with no
# --checkpoint, so all three arms were bit-identical baselines.
#
# The harvest chain left five intermediate checkpoints, one per adaptation
# episode, so the question can be answered without harvesting anything new:
# does the memory get BETTER as it accumulates experience?
#
#   round_00  39 entries,   5 nonzero, weight_sum 1.332, 1 episode
#   round_01  83 entries,  10 nonzero, weight_sum 2.664, 2 episodes
#   round_02 130 entries,  15 nonzero, weight_sum 3.996, 3 episodes
#   round_03 179 entries,  20 nonzero, weight_sum 5.328, 4 episodes
#   round_04 227 entries,  25 nonzero, weight_sum 6.660, 5 episodes
#
# Note the accumulation is EXACTLY linear -- every round contributes exactly 5
# nonzero entries carrying exactly the same total weight.  The credit rule never
# rejects an episode and never differentiates between them; it is a fixed
# geometric window over the last 5 steps, not a selection mechanism.  So this job
# is also a test of whether that matters: if outcomes are flat across rounds,
# "self-evolution" on this design is accumulation without improvement.
#
# READING:
#   monotone improvement -> self-evolution works, this is the paper's main figure
#   flat                 -> more experience buys nothing; criterion 3 fails and
#                           the credit rule, not the correction channel, is next
#   non-monotone         -> later episodes interfere with earlier ones
#
# OFFSETS.  The 12 offsets v8_seedpin already ran `--arm base` on THIS host, so
# the paired base reference costs nothing and is guaranteed same-hardware.
#   base-success: 3 15 19 25 35 46      base-failure: 0 26 29 36 42 43
# CAVEAT to report alongside: offsets 0 and 26 are ADAPTATION offsets -- the
# memory was harvested from them -- so they are training data and must be broken
# out separately, never pooled into the held-out fix rate.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

OFFSETS=(3 15 19 25 35 46 0 26 29 36 42 43)
NOFF=12
ROUNDS=(round_00_off0 round_01_off5 round_02_off10 round_03_off26 round_04_off27)

i=$((SGE_TASK_ID - 1))
RD=${ROUNDS[$((i / NOFF))]}
OFF=${OFFSETS[$((i % NOFF))]}

OUT=$LR/v9_rounds/$RD/off$OFF
mkdir -p "$OUT" $LR/v9_rounds/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="rounds_${RD}_off${OFF}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_RELATION_MEMORY_FEATURE_KEYS="cost_pair_min_distance,cost_pair_delta_xy[0],cost_pair_delta_xy[1],eef_hazard_distance,eef_target_distance,episode_step_index"
export SE_VLA_RELATION_MEMORY_AUTHORITY_GAIN=1.0
export SE_VLA_RELATION_MEMORY_MAGNITUDE_MODE=replay   # the locked method, unchanged
unset SE_VLA_RELATION_MEMORY_ABSTAIN_DISTANCE || true
unset SE_VLA_RELATION_MEMORY_RISK_GATE || true

CK=$LR/v7_l1t2/newkey-h5/$RD/settled_checkpoint.json
[ -s "$CK" ] || { echo "missing checkpoint $CK"; exit 1; }

echo "=== v9_rounds $RD off=$OFF ckpt=$CK on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm pruned_memory --checkpoint "$CK" \
  --port-base $((40000 + 25 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
echo "ROUNDS_EXIT=$? round=$RD off=$OFF"
