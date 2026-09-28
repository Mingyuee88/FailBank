#!/usr/bin/env bash
#$ -N v9_power
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 2
#$ -t 1-58
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v9_power/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v9_power/batch/j.$JOB_ID.$TASK_ID.out
#
# v9_power -- turn "we tried everything and nothing worked" into a claim with a
# denominator and a ceiling.  Two things are missing from the negative result:
#
# POWER.  Every v9_sched arm scored 0 fixes, but on only 4 base-failure offsets.
# A method with a true 25% fix rate shows 0/4 about a third of the time, so 0/4
# alone proves little.  v8_clean already gives replay 0/11 on this host's full
# 47-offset set; the best new arm (schedule, no abstain -- the only arm that beat
# replay on CC, -19.0% vs -16.2%, with obj_travel intact at 0.94x) deserves the
# same denominator before it is written off.
#
# CEILING.  The sharper question is not "does the memory fix things" but "is
# there anything to fix".  The shield saved 5/5 adaptation episodes (SR 0->1,
# polCC 348->0), but those were the offsets it was run on; the memory is judged
# on different ones.  Running the shield itself on the SAME held-out base-failure
# offsets gives the upper bound the distillation is being measured against.
#
#   shield fixes k/11, memory fixes 0/11  ->  the gap is the paper's result, and
#                                             it is a mechanism claim, not a
#                                             tuning failure
#   shield also fixes 0/11                ->  these offsets are not shield-fixable
#                                             at all, the whole L1t2 fix-rate
#                                             framing is void, and the negative
#                                             result says nothing about the method
#
# The second outcome would invalidate a lot of today's reasoning, which is
# exactly why it has to be run rather than assumed.
#
# Arms:
#   1-47   schedule / no abstain, full 47-offset census (11 base failures here)
#   48-58  shield ON, no memory, on the 11 base-failure offsets  <-- the ceiling
#
# Same host as v8_clean so base comes free and no hardware term enters.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

ALL=(0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47)
# base failures on HOST_A, read off v8_clean/base (note off35 fails on L40S
# and succeeds on A6000 -- this list is host-specific by construction)
FAILS=(0 5 10 26 27 29 35 36 47 42 43)

i=$((SGE_TASK_ID - 1))
if [ $i -lt 47 ]; then MODE=schedule; OFF=${ALL[$i]}; else MODE=shield; OFF=${FAILS[$((i - 47))]}; fi

OUT=$LR/v9_power/$MODE/off$OFF
mkdir -p "$OUT" $LR/v9_power/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="power_${MODE}_off${OFF}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
PORT=$((52000 + 25 * SGE_TASK_ID))

if [ "$MODE" = shield ]; then
  # frozen policy + the CBF shield, no memory at all. Verified recipe from the
  # harvest manifest: alpha 3 / eef 0.03 / oracle 0.04 / margin 0.02.
  export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
  export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04
  export SE_VLA_AEGIS_MARGIN=0.02 SE_VLA_AEGIS_MAX_TRANSLATION=1.0 SE_VLA_AEGIS_OVERRIDE_EPS=1e-6
  export SE_VLA_SRD_COOLDOWN_STEPS=8 SE_VLA_SRD_DUTY_CAP=0.15
  export SE_VLA_SRD_TTC_MIN_STEPS=1 SE_VLA_SRD_TTC_MAX_STEPS=8
  export SE_VLA_SRD_RECOVERY_HORIZON_STEPS=12 SE_VLA_SRD_MIN_RECOVERY_SEPARATION_M=0.01
  export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
  echo "=== v9_power SHIELD off=$OFF on $(hostname) ==="
  PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
    roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
    --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
    --replan-steps 1 --trials 1 --arm base \
    --port-base $PORT --array-task-id 1 --server-attempts 3
  echo "POWER_EXIT=$? mode=$MODE off=$OFF"; exit 0
fi

# schedule arm: shield off, memory read-only, live-gap magnitude
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_RELATION_MEMORY_FEATURE_KEYS="cost_pair_min_distance,cost_pair_delta_xy[0],cost_pair_delta_xy[1],eef_hazard_distance,eef_target_distance,episode_step_index"
export SE_VLA_RELATION_MEMORY_AUTHORITY_GAIN=1.0
export SE_VLA_RELATION_MEMORY_MAGNITUDE_MODE=schedule
export SE_VLA_RELATION_MEMORY_MAGNITUDE_SLOPE_SIGN=1.0
unset SE_VLA_RELATION_MEMORY_ABSTAIN_DISTANCE || true
unset SE_VLA_RELATION_MEMORY_RISK_GATE || true

CK=$(ls -d $LR/v7_l1t2/newkey-h5/round_*/settled_checkpoint.json 2>/dev/null | tail -1)
[ -n "$CK" ] || { echo "no settled checkpoint"; exit 1; }
echo "=== v9_power SCHEDULE off=$OFF ckpt=$CK on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm pruned_memory --checkpoint "$CK" \
  --port-base $PORT --array-task-id 1 --server-attempts 3
echo "POWER_EXIT=$? mode=$MODE off=$OFF"
