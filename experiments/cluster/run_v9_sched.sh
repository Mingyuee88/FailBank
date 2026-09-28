#!/usr/bin/env bash
#$ -N v9_sched
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 2
#$ -t 1-90
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v9_sched/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v9_sched/batch/j.$JOB_ID.$TASK_ID.out
#
# v9_sched -- does restoring the shield's GAIN SCHEDULE do what more authority could not?
#
# WHAT THE MEASUREMENTS SAY.  Comparing the 227 entries the shield wrote during
# adaptation against the 661 deployment steps that replayed them:
#
#                              corr(|d|, cost_pair_min_distance)   mean |d|
#     shield at write time              -0.252                      0.1942
#     memory replay, gain=1             +0.217                      0.0030
#     memory replay, gain=30            +0.267                      0.0852
#
# A barrier controller pushes harder as the gap closes, so the shield's negative
# correlation IS the control law and the replay's positive one is that law
# inverted.  The inversion is structural: the retrieval key puts bandwidth 1.0 on
# episode_step_index against 0.06 on every geometric dimension, so lookup is
# dominated by WHEN a step happened, not HOW DANGEROUS it is.  On top of that the
# magnitude collapses 65x (confidence_c0=1.0 against 0.25^age weights).
#
# The v8_auth sweep already showed amplification cannot repair this: 27x more
# magnitude at the actuator left the fix count pinned at 1/5 and began costing
# retention.  Amplifying an inverted law only makes it wronger.
#
# THE CHANGE.  magnitude_mode="schedule": retrieval still decides WHETHER to act
# (same keys, same kernel, same top_k, same abstain -- hit_rate must be unchanged)
# and the magnitude is recomputed from the LIVE gap using log|d| = a + b*gap fitted
# to the shield's own writes.  On this checkpoint the fit is
# a=-0.2584, b=-8.4673 (227 entries, slope required negative or the run aborts).
#
# WHY THE DUTY AXIS.  At the gaps deployment actually fires at (p50 = 0.116 m) the
# schedule asks for 17.75 mm/step, against the shield's own mean of 9.7 mm/step.
# That is not a mis-fit: the shield fired only on trigger, while the memory fires
# on 29% of all steps, so shield-sized corrections at a 29% duty is far MORE
# intervention than the shield ever applied.  Restoring the duty cycle is part of
# restoring the controller, so abstain is crossed with the magnitude mode rather
# than tuned after the fact.
#
# ARMS (6 x 15 offsets). base and replay/anone come free from v8_clean on THIS
# same host (47 offsets each, identical config), so only the new cells run here;
# replay/anone is kept in-experiment anyway as an end-to-end regression check that
# the patch left the locked path byte-identical.  The flipped arm is the one that makes this falsifiable:
# it moves the same magnitudes around with the barrier's sign reversed, so a win
# for "schedule" that also shows up in "schedule_flip" is just authority_gain
# wearing a different hat -- and authority has already been falsified once.
#
#   (base)               reused from v8_clean/base on HOST_A
#   replay      / a=none the current method (known: dSR 0.000 over the full 47)
#   schedule    / a=none live-gap magnitude, barrier sign
#   sched_flip  / a=none live-gap magnitude, sign reversed          <-- control
#   replay      / a=0.7  current method at shield-like duty
#   schedule    / a=0.7  the full hypothesis
#   sched_flip  / a=0.7  its control
#
# SOURCE AUDIT, reported alongside the outcome (repo rule: self-invented config
# silently no-ops).  From telemetry, per arm:
#   (a) hit_rate must be IDENTICAL between replay and schedule at the same abstain
#       -- retrieval is untouched, only magnitude moves.  If it differs, the arms
#       are not comparable and the cell is void.
#   (b) corr(|d|, cost_pair_min_distance) must be ~+0.22 for replay, clearly
#       negative for schedule, more positive for sched_flip.  If all three match,
#       the factor did not take effect and the results carry no information.
#
# Everything is pinned to ONE host: swapping GPU model alone moves polCC on 11/12
# offsets (+17.5%) and flips SR on 1/12, which is larger than any effect here.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

# 5 base-failure (fix rate) + 10 base-success (retention)
OFFSETS=(29 36 38 42 43 1 2 4 6 11 17 20 24 31 44)
NOFF=15
ARMS=(replay_anone schedule_anone schedflip_anone replay_a07 schedule_a07 schedflip_a07)

i=$((SGE_TASK_ID - 1))
ARM=${ARMS[$((i / NOFF))]}
OFF=${OFFSETS[$((i % NOFF))]}

OUT=$LR/v9_sched/$ARM/off$OFF
mkdir -p "$OUT" $LR/v9_sched/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="v9_${ARM}_off${OFF}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
# deployment: shield OFF, memory read-only, no writes, no credit
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
PORT=$((12000 + 25 * SGE_TASK_ID))

if [ "$ARM" = base ]; then
  echo "=== v9 base off=$OFF on $(hostname) ==="
  PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
    roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
    --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
    --replan-steps 1 --trials 1 --arm base \
    --port-base $PORT --array-task-id 1 --server-attempts 3
  echo "V9_EXIT=$? arm=$ARM off=$OFF"; exit 0
fi

# byte-for-byte the retrieval key of the selected arm; retrieval is held FIXED
# across every memory arm here.
export SE_VLA_RELATION_MEMORY_FEATURE_KEYS="cost_pair_min_distance,cost_pair_delta_xy[0],cost_pair_delta_xy[1],eef_hazard_distance,eef_target_distance,episode_step_index"
export SE_VLA_RELATION_MEMORY_AUTHORITY_GAIN=1.0
unset SE_VLA_RELATION_MEMORY_RISK_GATE || true

case "$ARM" in
  *_anone) unset SE_VLA_RELATION_MEMORY_ABSTAIN_DISTANCE || true ;;
  *_a07)   export SE_VLA_RELATION_MEMORY_ABSTAIN_DISTANCE=0.7 ;;
esac
case "$ARM" in
  replay_*)    export SE_VLA_RELATION_MEMORY_MAGNITUDE_MODE=replay ;;
  schedule_*)  export SE_VLA_RELATION_MEMORY_MAGNITUDE_MODE=schedule
               export SE_VLA_RELATION_MEMORY_MAGNITUDE_SLOPE_SIGN=1.0 ;;
  schedflip_*) export SE_VLA_RELATION_MEMORY_MAGNITUDE_MODE=schedule
               export SE_VLA_RELATION_MEMORY_MAGNITUDE_SLOPE_SIGN=-1.0 ;;
esac

CK=$(ls -d $LR/v7_l1t2/newkey-h5/round_*/settled_checkpoint.json 2>/dev/null | tail -1)
[ -n "$CK" ] || { echo "no settled checkpoint"; exit 1; }

echo "=== v9 arm=$ARM off=$OFF mode=${SE_VLA_RELATION_MEMORY_MAGNITUDE_MODE} "\
"sign=${SE_VLA_RELATION_MEMORY_MAGNITUDE_SLOPE_SIGN:-n/a} "\
"abstain=${SE_VLA_RELATION_MEMORY_ABSTAIN_DISTANCE:-none} on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm pruned_memory --checkpoint "$CK" \
  --port-base $PORT --array-task-id 1 --server-attempts 3
echo "V9_EXIT=$? arm=$ARM off=$OFF"
