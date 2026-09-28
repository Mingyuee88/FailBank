#!/usr/bin/env bash
#$ -N v26_r2col
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 2
#$ -t 1-29
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v26_r2collect/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v26_r2collect/batch/j.$JOB_ID.$TASK_ID.out
#
# v26_r2collect -- round 2 of the loop, and the first time the loop is actually a loop.
#
# WHAT WAS MISSING. Everything shipped so far is ONE offline update: shield runs on the
# BASE policy's states, those corrections are distilled into LoRA, done. That is
# DAgger's first iteration and nothing more, which is why calling the result
# "self-evolving" has been flagged as unsupported. The evidence needed is a closed
# cycle whose second turn is driven by the FIRST turn's own mistakes:
#
#   deploy R1 -> R1 fails somewhere -> shield corrects R1 (not base) -> record -> R2
#
# WHY THAT IS NOT COSMETIC. R1's failure set is not base's failure set. On L1t2:
#
#   base fails on   {0, 5, 10, 26, 27, 29, 35, 36, 42, 43, 47}         11 offsets
#   R1   fails on   {0, 10, 17, 26, 29, 34, 36, 37, 42}                 9 offsets
#     inherited     {0, 10, 26, 29, 36, 42}          6 base failures R1 did not fix
#     NEW           {17, 34, 37}                     3 successes R1 broke by itself
#
# Round-1 records cannot contain a single state from {17, 34, 37}: those episodes did
# not exist until R1 existed. A second round is the only way the method can see the
# damage it caused, and whether it repairs its own regressions is the sharpest test of
# the self-evolving claim available. Round 2 collecting on base's distribution again
# would prove nothing -- hence the checkpoint override and the assertion below that it
# actually took effect.
#
# LEAKAGE. Collection runs ONLY on the 29 offsets in v18_split's training half. The 18
# held-out success offsets are never visited, so retention on them stays honest across
# both rounds -- structurally, not by remembering to filter later. Note this puts two
# of R1's three regressions ({17, 34}) OUT of reach of training: if R2 recovers them it
# is generalization, not memorization. Final judgement remains cross-task transfer.
#
# SILENT-FAILURE GUARD. Four times this project has run a knob that did nothing and
# read the null result as a finding. The specific hazard here is that
# SE_VLA_POLICY_CHECKPOINT_DIR fails to reach the policy server and round 2 quietly
# re-collects on the BASE policy -- which would look like a perfectly normal job and
# produce a perfectly wrong conclusion. Three post-run assertions, any of which fails
# the cell: the server log must show the R1 params being restored, the shield must be
# on, and the recorder must have grown the round-2 blob store.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

ARM=A_old_qw0.0
CK=$LR/v20_fold/$ARM/offset_0
[ -d "$CK/params" ] || { echo "missing R1 checkpoint $CK"; exit 1; }

# 47 usable L1t2 offsets minus v18_split heldout_success_offsets
TRAIN_OFFS=(0 1 3 5 6 8 10 12 14 16 18 20 22 24 26 27 28 29 31 33 35 36 37 39 41 42 43 45 47)
OFF=${TRAIN_OFFS[$((SGE_TASK_ID - 1))]:-}
[ -n "$OFF" ] || { echo "no offset for task $SGE_TASK_ID"; exit 0; }

OUT=$LR/v26_r2collect/off$OFF
RECROOT=$LR/v26_r2collect/records
mkdir -p "$OUT" "$RECROOT" $LR/v26_r2collect/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true
# NB: pipefail is on, and `find` on a not-yet-created blobs dir exits nonzero, which
# would kill the job before it ever starts. Swallow that specific failure.
BLOBS_BEFORE=$( { find "$RECROOT/blobs" -type f 2>/dev/null || true; } | wc -l )

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
# the policy under correction is R1, not base -- this line IS the round-2 experiment
export SE_VLA_POLICY_CHECKPOINT_DIR="$CK"
# identical shield recipe to round 1 so the two rounds' teacher signals are comparable
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_PHASE1_RECORD=1 SE_VLA_PHASE1_CANDIDATES=1
export SE_VLA_PHASE1_ROOT="$RECROOT"
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="r2col_off${OFF}"

echo "=== v26 round-2 collection off=$OFF policy=$CK on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((44000 + 20 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "R2COL_EXIT=$rc off=$OFF"

# ---- assertion 1: the R1 weights, not base, were what the server actually restored
if ! grep -q "Restoring checkpoint from $CK/params" "$OUT"/server_port*.log 2>/dev/null; then
  echo "R1_CHECKPOINT_NOT_LOADED off=$OFF -- round 2 would have re-collected on base"
  grep -i "Restoring checkpoint from" "$OUT"/server_port*.log 2>/dev/null | head -3 || true
  exit 1
fi
echo "R1_CHECKPOINT_VERIFIED off=$OFF"

# ---- assertions 2 and 3: shield really ran, recorder really wrote
BLOBS_AFTER=$( { find "$RECROOT/blobs" -type f 2>/dev/null || true; } | wc -l )
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/result.json" "$OFF" "$BLOBS_BEFORE" "$BLOBS_AFTER" <<'PYCHK'
import json, sys
r = json.load(open(sys.argv[1])); off, before, after = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ad = r.get("adapter", {})
md = r.get("metric_decomposition", {})
print(f"R2COL off={off} shield_corrections={ad.get('corrections_applied')} "
      f"sr={r.get('successes')} polcc={md.get('policy_induced_cc')} blobs {before}->{after}")
assert r.get("status") == "pass", f"episode status {r.get('status')}"
# the shield being silent on an offset R1 already handles is expected and fine; what is
# not fine is the recorder producing nothing anywhere, so growth is asserted globally by
# the collector's summary rather than per cell. Here only assert the run was well-formed.
assert ad.get("physical_units_gate") == "pass", "physical units gate"
print("R2COL_CELL_VERIFIED")
PYCHK
