#!/usr/bin/env bash
#$ -N v10_ctrl
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 2
#$ -t 1-141
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v10_ctrl/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v10_ctrl/batch/j.$JOB_ID.$TASK_ID.out
#
# v10_ctrl -- the three controls an adversarial review said the 08-10 run was missing.
#
# Codex reviewed the full evidence chain and made three objections that each map
# to a cheap, decisive experiment. All three run here, full 47-offset census, on
# the same pinned host as v8_clean and v9_power so base, replay and schedule are
# all directly paired with no hardware term.
#
# (1) THE FLIPPED-SLOPE CONTROL WAS CONFOUNDED.
#     schedflip reversed the slope but kept the fitted intercept, so its mean
#     |delta| was 1.4151 against schedule's 0.2182 -- 6.5x larger. It therefore
#     could not separate "the schedule has the right shape" from "the pushes are
#     simply bigger", which was its only job. Two magnitude-matched controls:
#
#       const  a=-1.5225 b=0        constant |delta| = 0.2182, matched to the
#                                   schedule arm's measured deployment mean, with
#                                   NO dependence on the gap at all.
#       flipm  a=-2.2224 b=+8.4673  same magnitude as schedule at the median
#                                   deployment gap (0.116 m), opposite slope.
#
#     schedule beating const  => the gap-dependence itself carries the benefit
#     schedule == const       => only the magnitude mattered, and "restoring the
#                                barrier's control law" is not the mechanism
#
# (2) "episode_step_index DOMINATES THE KEY" WAS NEVER ABLATED.
#     The claim that retrieval matches on WHEN rather than HOW DANGEROUS came from
#     comparing bandwidths (1.0 on the step index against 0.06 on every geometric
#     dimension), not from an experiment.
#
#       nostep  the same schedule arm with episode_step_index removed from the key
#
#     If the root-cause claim is right this should change retrieval markedly and,
#     if the diagnosis is actionable, improve outcomes. If it changes nothing, the
#     bandwidth argument was numerology.
#
# (3) POWER. Every arm runs all 47 offsets (11 base failures, 6 of them held out
#     from the harvest) rather than the 15-offset screen, so each arm gets the
#     largest denominator this arena can provide. That denominator is still small:
#     a 1/6 fix rate carries a 95% Wilson interval of 3.0%-56.4%, and reaching 80%
#     power at the observed effect size needs roughly 39 base-failure offsets. So
#     these results are EXPLORATORY by construction and are reported as such --
#     they cannot confirm anything on this arena, only rank hypotheses for a
#     properly preregistered multi-arena run.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

OFFSETS=(0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47)
NOFF=47
ARMS=(const flipm nostep)
i=$((SGE_TASK_ID - 1))
ARM=${ARMS[$((i / NOFF))]}
OFF=${OFFSETS[$((i % NOFF))]}

OUT=$LR/v10_ctrl/$ARM/off$OFF
mkdir -p "$OUT" $LR/v10_ctrl/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="v10_${ARM}_off${OFF}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_RELATION_MEMORY_AUTHORITY_GAIN=1.0
export SE_VLA_RELATION_MEMORY_MAGNITUDE_MODE=schedule
unset SE_VLA_RELATION_MEMORY_ABSTAIN_DISTANCE || true
unset SE_VLA_RELATION_MEMORY_RISK_GATE || true

FULLKEY="cost_pair_min_distance,cost_pair_delta_xy[0],cost_pair_delta_xy[1],eef_hazard_distance,eef_target_distance,episode_step_index"
case "$ARM" in
  const)  export SE_VLA_RELATION_MEMORY_FEATURE_KEYS="$FULLKEY"
          export SE_VLA_RELATION_MEMORY_MAGNITUDE_COEFFS="-1.5225,0.0" ;;
  flipm)  export SE_VLA_RELATION_MEMORY_FEATURE_KEYS="$FULLKEY"
          export SE_VLA_RELATION_MEMORY_MAGNITUDE_COEFFS="-2.2224,8.4673" ;;
  nostep) export SE_VLA_RELATION_MEMORY_FEATURE_KEYS="cost_pair_min_distance,cost_pair_delta_xy[0],cost_pair_delta_xy[1],eef_hazard_distance,eef_target_distance"
          unset SE_VLA_RELATION_MEMORY_MAGNITUDE_COEFFS || true ;;
esac

CK=$(ls -d $LR/v7_l1t2/newkey-h5/round_*/settled_checkpoint.json 2>/dev/null | tail -1)
[ -n "$CK" ] || { echo "no settled checkpoint"; exit 1; }
echo "=== v10 arm=$ARM off=$OFF coeffs=${SE_VLA_RELATION_MEMORY_MAGNITUDE_COEFFS:-fitted} on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm pruned_memory --checkpoint "$CK" \
  --port-base $((8000 + 25 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
echo "V10_EXIT=$? arm=$ARM off=$OFF"
