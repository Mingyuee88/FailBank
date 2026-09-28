#!/usr/bin/env bash
#$ -N v8_auth
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-135
#$ -tc 10
#$ -o ${WORK_ROOT}/lora_dagger/v8_auth/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v8_auth/batch/j.$JOB_ID.$TASK_ID.out
#
# v8_auth -- authority x selectivity dose-response for the relation memory.
#
# WHY.  Measured over the 42 deployment episodes of the winning arm
# (newkey-h5, no gate), from metadata.runtime_status in shield_steps.jsonl:
#
#     matched (fires) on 1450/5057 steps = 28.7% of all steps
#     applied |delta|  mean 0.0044  median 0.0017  p95 0.0142   [action units]
#     confidence sum   median 0.031  ->  confidence_scale = w/(w+c0) ~= 0.030
#     dual_gain        mean 0.79
#
# Arena maps action +-1 to +-0.05 m, so the mean correction actually reaching the
# actuator is 0.0044 * 0.05 m = 0.22 mm per step, against a shield safety margin
# of 20 mm.  The method has been running at roughly 3% authority, and it fires on
# 29% of steps.  That is the wrong corner of the design space in both coordinates
# and it fully explains the measured outcome on 42 offsets: dSR = 0.000,
# dCC = -3.7% (NOT the +2.3pt / -20% in the 08-10 handoff, which compared against
# a mis-stated base of 35/47; the true base is 37/47 with mean polCC 57.3).
#
# HYPOTHESIS.  Fire rarely, with real authority.  Two knobs, both verified to be
# re-applied after checkpoint load in openvla_runtime.py (the guard that silently
# neutered the first authority probe is fixed for authority_gain and was already
# correct for abstain_distance):
#
#   AUTHORITY_GAIN   3 / 10 / 30      3x, 10x, 30x the correction magnitude.
#                                     30 restores roughly the magnitude the
#                                     shield itself applied when it wrote them.
#   ABSTAIN_DISTANCE none / 1.2 / 0.7 empirical best_distance quantiles give
#                                     hit_rate 0.287 / 0.122 / 0.036.
#
# SOURCE AUDIT (rule from HANDOFF 7.1 -- self-invented config silently no-ops).
# Two monotonicity canaries, both read back from telemetry by the analysis:
#   (a) mean applied |delta| must scale ~linearly with AUTHORITY_GAIN
#   (b) hit_rate must decrease monotonically as ABSTAIN_DISTANCE tightens
# If either flat-lines across levels, that factor did not take effect and the
# cell's result carries no information.
#
# OFFSETS.  15 already-burned offsets: the 5 held-out base-failure probes
# (29,36,38,42,43 -> fix rate) and the 10 base-success selection probes
# (1,2,4,6,11,17,20,24,31,44 -> retention).  The 27 never-probed base-success
# offsets are deliberately NOT touched here so they stay clean for confirming
# whichever cell wins.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

GAINS=(3 10 30)
ABSTS=(none 1.2 0.7)
OFFSETS=(29 36 38 42 43 1 2 4 6 11 17 20 24 31 44)
NOFF=15

i=$((SGE_TASK_ID - 1))
cell=$((i / NOFF))
OFF=${OFFSETS[$((i % NOFF))]}
G=${GAINS[$((cell / 3))]}
A=${ABSTS[$((cell % 3))]}

ARM_DIR=newkey-h5
CELL="g${G}_a${A}"
OUT=$LR/v8_auth/$CELL/off$OFF
mkdir -p "$OUT" $LR/v8_auth/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP existing $OUT"; exit 0; fi

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="auth_${CELL}_off${OFF}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
# deployment: shield OFF, memory read-only, no writes, no credit
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
# byte-for-byte the key of the winning arm
export SE_VLA_RELATION_MEMORY_FEATURE_KEYS="cost_pair_min_distance,cost_pair_delta_xy[0],cost_pair_delta_xy[1],eef_hazard_distance,eef_target_distance,episode_step_index"
# the two swept knobs
export SE_VLA_RELATION_MEMORY_AUTHORITY_GAIN="$G"
if [ "$A" = none ]; then unset SE_VLA_RELATION_MEMORY_ABSTAIN_DISTANCE
else export SE_VLA_RELATION_MEMORY_ABSTAIN_DISTANCE="$A"; fi
# the blunt post-hoc risk gate stays OFF; abstain_distance is the selectivity
# control being tested and mixing the two is what made the 08-10 gate arms
# uninterpretable.
unset SE_VLA_RELATION_MEMORY_RISK_GATE || true

CK=$(ls -d $LR/v7_l1t2/$ARM_DIR/round_*/settled_checkpoint.json 2>/dev/null | tail -1)
[ -n "$CK" ] || { echo "no settled checkpoint for $ARM_DIR"; exit 1; }

echo "=== v8_auth cell=$CELL gain=$G abstain=$A off=$OFF ckpt=$CK ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm pruned_memory --checkpoint "$CK" \
  --port-base $((50000 + 30 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
echo "AUTH_EXIT=$? cell=$CELL off=$OFF"
