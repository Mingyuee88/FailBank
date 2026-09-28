#!/usr/bin/env bash
#$ -N v23_shxfer
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_B
#$ -pe smp 2
#$ -t 1-150
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v23_shxfer/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v23_shxfer/batch/j.$JOB_ID.$TASK_ID.out
#
# v23_shxfer -- the SHIELD's own transfer profile, which the misalignment claim needs.
#
# WHY THIS ARM. Assembling the shield's own full 47-offset profile for the first
# time (v9_power/shield on the 11 base-failure offsets + v16_negcollect on the 36
# base-success ones, both pinned to the same host, same verified recipe) shows the
# misalignment this project was founded on, quantified:
#
#     AEGIS shield   fix 10/11 = 90.9%   retention 27/36 = 75.0%   dCC -91.0%   NET SR +1
#     LoRA arm A     fix  5/11 = 45.5%   retention 33/36 = 91.7%   dCC -26.4%   NET SR +2
#
# The shield drives cost down 91% and nets a single episode, because it destroys 9
# of the 36 successes it touches. Low CC bought by breaking a quarter of the task
# is exactly "CC down, SR down". So the shield is the baseline's LOWER bound, not a
# ceiling, and the bar to clear is its NET, not its CC. Arm A already clears it on
# this arena (+2 vs +1, trade ratio 5:3 against 10:9) with no runtime device at all.
#
# WHAT IS UNTESTED. All of that is one arena. The external-memory line looked
# strongest on exactly this development domain -- geo5 reached fix 4/6 there -- and
# then destroyed half the successes on transfer (retention 47.4%, dSR -0.333). A
# within-task win is not evidence until it survives unseen objects.
#
# THE TEST. The stock frozen policy with AEGIS running, on the same 150 offsets arm
# A was just measured on. Arm A transferred well -- fix 23/34 = 67.6%, retention
# 84.5%, net SR +5, no task significantly harmed. The open question is whether the
# runtime shield does better or worse on objects whose gripper-to-object offset
# differs from mango's: its barrier sits on the gripper while the cost predicate
# sits on the manipulated object, and on L1t2 that already cost it 9 of 36
# successes for 10 repairs (net +1). Same pinned host, same base reference
# (v11_merged), so all three -- base, arm A, shield -- are paired per offset.
#
# READOUT, fixed before submission: per-task fix rate and retention rate with
# Wilson intervals, never netted and never pooled into a single binomial (episodes
# cluster by task -- pooling hid a task being significantly harmed in v13_xfer);
# paired dCC; obj_travel ratio flagged in both directions; exact McNemar per task.
#
# The comparison that matters is against the SHIELD's transfer profile, not against
# zero. That profile is not yet measured on these tasks and is the natural follow-up
# if this arm holds.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

TASKS=(1 3 4)
NOFF=50
i=$((SGE_TASK_ID - 1))
TID=${TASKS[$((i / NOFF))]}
OFF=$((i % NOFF))

OUT=$LR/v23_shxfer/L1t${TID}/off${OFF}
mkdir -p "$OUT" $LR/v23_shxfer/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="shxfer_L1t${TID}_off${OFF}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
# the SHIELD on the stock frozen policy: the comparison the misalignment claim needs
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04
export SE_VLA_AEGIS_MARGIN=0.02 SE_VLA_AEGIS_MAX_TRANSLATION=1.0 SE_VLA_AEGIS_OVERRIDE_EPS=1e-6
export SE_VLA_SRD_COOLDOWN_STEPS=8 SE_VLA_SRD_DUTY_CAP=0.15
export SE_VLA_SRD_TTC_MIN_STEPS=1 SE_VLA_SRD_TTC_MAX_STEPS=8
export SE_VLA_SRD_RECOVERY_HORIZON_STEPS=12 SE_VLA_SRD_MIN_RECOVERY_SEPARATION_M=0.01
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0

echo "=== v23 SHIELD L1t${TID} off=$OFF on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id "$TID" --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((38000 + 20 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
echo "SHXFER_EXIT=$? L1t$TID off=$OFF"
