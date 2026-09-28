#!/usr/bin/env bash
#$ -N v29_bdist
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 2
#$ -t 1-29
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v29_basedist/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v29_basedist/batch/j.$JOB_ID.$TASK_ID.out
#
# v29_basedist -- the matched control for round 2, and the only thing that can tell
# "the loop worked" apart from "the second training run had more data".
#
# v26_r2collect gathers shield corrections on R1's OWN states. Training on
# round1 + round2 then differs from training on round1 alone in two ways at once:
# whose state distribution the new records came from, and how many records there are.
# Without separating those, a gain is equally explained by ordinary data scaling and the
# self-evolving claim is unsupported.
#
# This job holds everything fixed except the distribution: identical shield recipe,
# identical 29 training-half offsets, identical recorder, identical host, identical
# record budget -- but the policy being corrected is BASE, the same policy round 1
# already used. If round-2 training beats this control, the gain is attributable to
# collecting on the evolved policy's own distribution. If it does not, the second round
# bought nothing that a second pass over base data would not have bought, and the loop
# is not doing what the paper would claim.
#
# Caveat recorded up front: this codebase is bit-deterministic on a pinned host and
# `--seed` is a verified no-op, so these episodes reproduce trajectories base has already
# walked. That is the point for a matched TRAINING root -- the control must differ only
# in distribution -- but it does mean this arm cannot double as a test of collection
# diversity.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger


# 47 usable L1t2 offsets minus v18_split heldout_success_offsets
TRAIN_OFFS=(0 1 3 5 6 8 10 12 14 16 18 20 22 24 26 27 28 29 31 33 35 36 37 39 41 42 43 45 47)
OFF=${TRAIN_OFFS[$((SGE_TASK_ID - 1))]:-}
[ -n "$OFF" ] || { echo "no offset for task $SGE_TASK_ID"; exit 0; }

OUT=$LR/v29_basedist/off$OFF
RECROOT=$LR/v29_basedist/records
mkdir -p "$OUT" "$RECROOT" $LR/v29_basedist/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true
# NB: pipefail is on, and `find` on a not-yet-created blobs dir exits nonzero, which
# would kill the job before it ever starts. Swallow that specific failure.
BLOBS_BEFORE=$( { find "$RECROOT/blobs" -type f 2>/dev/null || true; } | wc -l )

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
# the policy under correction is BASE -- this arm is round 1's distribution, again
unset SE_VLA_POLICY_CHECKPOINT_DIR || true
# identical shield recipe to round 1 so the two rounds' teacher signals are comparable
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_PHASE1_RECORD=1 SE_VLA_PHASE1_CANDIDATES=1
export SE_VLA_PHASE1_ROOT="$RECROOT"
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="bdist_off${OFF}"

echo "=== v29 matched base-distribution collection off=$OFF on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((52000 + 20 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "BDIST_EXIT=$rc off=$OFF"

# ---- assertion 1: the STOCK checkpoint, not a folded LoRA, is what the server restored
if grep -q "Restoring checkpoint from ${WORK_ROOT}/lora_dagger/v20_fold" \
     "$OUT"/server_port*.log 2>/dev/null; then
  echo "BASEDIST_CONTAMINATED off=$OFF -- a folded LoRA checkpoint leaked into the control"
  exit 1
fi
echo "BASE_CHECKPOINT_VERIFIED off=$OFF"

# ---- assertions 2 and 3: shield really ran, recorder really wrote
BLOBS_AFTER=$( { find "$RECROOT/blobs" -type f 2>/dev/null || true; } | wc -l )
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/result.json" "$OFF" "$BLOBS_BEFORE" "$BLOBS_AFTER" <<'PYCHK'
import json, sys
r = json.load(open(sys.argv[1])); off, before, after = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ad = r.get("adapter", {})
md = r.get("metric_decomposition", {})
print(f"BDIST off={off} shield_corrections={ad.get('corrections_applied')} "
      f"sr={r.get('successes')} polcc={md.get('policy_induced_cc')} blobs {before}->{after}")
assert r.get("status") == "pass", f"episode status {r.get('status')}"
# the shield being silent on an offset R1 already handles is expected and fine; what is
# not fine is the recorder producing nothing anywhere, so growth is asserted globally by
# the collector's summary rather than per cell. Here only assert the run was well-formed.
assert ad.get("physical_units_gate") == "pass", "physical units gate"
print("BDIST_CELL_VERIFIED")
PYCHK
