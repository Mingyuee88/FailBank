#!/usr/bin/env bash
#$ -N v26_cens
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -pe smp 2
#$ -t 1-250
#$ -tc 3
#$ -o ${WORK_ROOT}/lora_dagger/v26_census/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v26_census/batch/j.$JOB_ID.$TASK_ID.out
#
# v26_census -- base-policy census on three safety suites never touched by this project.
#
# WHY. The whole claim currently rests on one arena family (safety_static_obstacles L1,
# five pick-and-place tasks that differ only in which fruit is on the table). The
# transfer set has 34 base-failure offsets and the paired McNemar on the method comes
# out p=0.533; by the observed effect size, 80% power needs roughly 39+ failure
# offsets, and they have to be independent arenas rather than the same template with a
# different object. Just adding offsets inside L1 cannot buy that.
#
# WHAT IS NEW HERE. These three suites carry different safety SEMANTICS, which is what
# makes them a real test of the gripper-barrier / object-cost misalignment rather than
# more of the same:
#   safety_hazard_avoidance   cost attaches to a lit candle / hot stove -- a hazard the
#                             barrier has never been fitted to
#   safety_state_preservation cost is about not disturbing what is already placed
#   safety_cautious_grasp     knives, scissors, forks -- object geometry unlike fruit
#
# WHAT THIS JOB DOES *NOT* DO. It measures the base policy only. Choosing an arena
# before looking at the base profile has burned this project repeatedly: L0 and L1t0
# have policy-induced cost identically 0 on their failures (no cost exists to remove),
# and L2t0/L2t1 sit at SR 0.06/0.00, where "did nothing" and "tried and failed" are the
# same number. An arena is admissible only if base SR lands in a middle band AND the
# failures carry nonzero policy-induced cost.
#
# PINNING. One suite per host, fixed for the life of the suite. GPU model alone moves
# policy-induced CC by 17.5% and flips SR on about 1 offset in 12, so a base list
# gathered on mixed hardware cannot be compared against anything measured later.
# Host is supplied on the qsub line; the assignment is recorded in host.txt per cell.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

: "${SUITE:?SUITE must be passed with qsub -v SUITE=...}"
# LEVEL is a variable because difficulty, not semantics, is what makes an arena
# unusable: safety_hazard_avoidance L1 is cost-rich (median failure polCC 357-397) but
# base solves 3 of 50, which is the zero-competence trap. The same suite at a lower
# level may sit in the band where both SR and cost carry signal.
LEVEL=${LEVEL:-1}
NTASK=5
i=$((SGE_TASK_ID - 1))
# Task varies FASTEST so that any prefix of the array is a balanced screen across all
# five tasks rather than an exhaustive sweep of task 0. The first run of this census
# spent an hour establishing that safety_hazard_avoidance t0 is a zero-competence trap
# (base 3/50) while learning nothing about t1-t4; ordering it this way would have
# answered the admissibility question for the whole suite in the same hour.
TID=$((i % NTASK))
OFF=$((i / NTASK))

OUT=$LR/v26_census/$SUITE/L${LEVEL}t${TID}/off${OFF}
mkdir -p "$OUT" $LR/v26_census/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="cens_${SUITE}_L${LEVEL}t${TID}_off${OFF}"
# run.py requires the three SRD distances even with the shield off
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
# pure base policy: no shield, no memory, no wrapper, stock checkpoint
unset SE_VLA_POLICY_CHECKPOINT_DIR || true
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0

echo "=== v26_census suite=$SUITE L${LEVEL}t${TID} off=$OFF on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite "$SUITE" \
  --task-level "$LEVEL" --task-id "$TID" --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((40000 + 20 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "CENSUS_EXIT=$rc suite=$SUITE L${LEVEL}t$TID off=$OFF"

# Post-run audit. A cell that produced no result.json, or one whose recorded identity
# does not match what was requested, is worse than a missing cell because it silently
# enters the base list. Fail loudly instead.
if [ -s "$OUT/result.json" ]; then
  external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/result.json" "$SUITE" "$TID" "$OFF" "$LEVEL" <<'PYCHK'
import json, sys
r = json.load(open(sys.argv[1])); suite, tid, off, lvl = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
ti = r.get("task_identity", {})
assert r.get("task_suite_name") == suite, f"suite mismatch {r.get('task_suite_name')} != {suite}"
assert int(r.get("task_id", -1)) == tid, f"task_id mismatch {r.get('task_id')} != {tid}"
assert int(r.get("init_state_offset", -1)) == off, f"offset mismatch {r.get('init_state_offset')} != {off}"
assert ti.get("identity_gate") == "pass", f"identity_gate {ti.get('identity_gate')}"
assert int(r.get("task_level", -1)) == lvl, f"level mismatch {r.get('task_level')} != {lvl}"
print(f"CENSUS_CELL_VERIFIED suite={suite} L{lvl}t{tid} off{off} "
      f"sr={r.get('successes')} polcc={r.get('metric_decomposition',{}).get('policy_induced_cc')} "
      f"bddl={ti.get('task_bddl_file')}")
PYCHK
else
  echo "CENSUS_CELL_EMPTY suite=$SUITE L${LEVEL}t$TID off$OFF"
fi
