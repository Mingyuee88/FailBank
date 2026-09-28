#!/usr/bin/env bash
#$ -N v16_neg
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 2
#$ -t 1-37
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v16_negcollect/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v16_negcollect/batch/j.$JOB_ID.$TASK_ID.out
#
# v16_negcollect -- the negative half of the teacher set, which has never existed.
#
# WHY. The LoRA distillation line reached fix rate 40% (18/45) with retention
# 88.2% (268/304), i.e. it repaired failures and destroyed 11.8% of the successes
# it should have left alone, for a whole-task net of about zero. The cause is
# visible in the collection manifest, not in the training:
#
#     phase1_collect.py  OFFSETS = [0,5,10,26,27,29,36,38,42,43]
#     derived/folds/offset_0/manifest.json
#         train_offsets = ["5","10","26","27","29","38","42","43"]
#         train_records = 2830, train_positive_teacher = 866
#
# Those ten offsets are exactly the ten base-FAILURE offsets. Every teacher record
# in the entire line comes from an episode the base policy was going to fail. The
# 37 base-SUCCESS offsets contribute nothing, so the model has never seen the
# state distribution of a successful grasp paired with a "do nothing" target. A
# model trained only on states that warrant a correction learns to correct
# everywhere -- which is precisely the failure mode the permutation control
# exposed in the memory arms today.
#
# The existing `quiet_weight` knob cannot fix this and its four-arm sweep
# correctly found nothing: it reweights the quiet STEPS INSIDE FAILING episodes,
# a different distribution from the states visited during successful ones.
#
# WHAT THIS COLLECTS. The same recorder, the same verified shield recipe
# (alpha 3 / eef 0.03 / oracle 0.04 / margin 0.02), run on the 37 base-success
# offsets. The shield will rarely fire there, which is the point: these episodes
# supply the negative examples -- "this state, no correction" -- that the teacher
# set has always lacked. Records land in a SEPARATE root so nothing existing is
# touched and the two halves can be mixed at any ratio afterwards.
#
# Pinned to one host like everything else since 2026-08-10: GPU model alone moves
# policy CC by 17.5% and flips SR on about 1 offset in 12.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

# the 37 base-success offsets on HOST_A, read off v8_clean/base
OKS=(1 2 3 4 6 7 8 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 28 30 31 32 33 34 37 38 39 40 41 44 45 46)
OFF=${OKS[$((SGE_TASK_ID - 1))]:-}
[ -n "$OFF" ] || { echo "no offset for task $SGE_TASK_ID"; exit 0; }

OUT=$LR/v16_negcollect/off$OFF
mkdir -p "$OUT" $LR/v16_negcollect/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
# shield ON so the recorder sees the same teacher signal as the positive half;
# on these offsets it should mostly decline to act, and that is the label we want
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_PHASE1_RECORD=1 SE_VLA_PHASE1_CANDIDATES=1
export SE_VLA_PHASE1_ROOT=$LR/v16_negcollect/records
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="neg_off${OFF}"

echo "=== v16 negative teacher collection off=$OFF on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((24000 + 20 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
echo "NEG_EXIT=$? off=$OFF"
