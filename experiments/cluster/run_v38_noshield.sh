#!/usr/bin/env bash
#$ -N v38_nosh
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 2
#$ -t 1-10
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v38_noshield/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v38_noshield/batch/j.$JOB_ID.$TASK_ID.out
#
# v38_noshield -- the control that decides what the method's active ingredient actually is.
#
# WHERE THIS COMES FROM. Tested on the endpoint this project originally registered --
# policy-induced cost, paired per offset, with the success-rate guard alongside -- the
# distillation line is POSITIVE: on L2 it removes 35-41% of policy-induced cost while
# success rate rises, and on hazard the reduction is significant (Wilcoxon p=0.0496 for the
# shield-corrected arm, p=0.0060 for SFT0). But SFT0 matches or beats the shield-corrected
# arm everywhere, which rules out the obvious explanation.
#
# SFT0 is not "no experience". Its targets are the base policy's own clipped actions, but
# its STATES come entirely from episodes the shield was steering. So the hypothesis that
# survives is:
#
#     the active ingredient is the state distribution the shield drags the policy through,
#     not the corrective actions it supplies.
#
# One piece of evidence already points that way: SFT0 draws from episodes where the shield
# fires constantly (base-FAILURE offsets) and reaches -41.4% cost at p=0.009 on L2, while
# SFTPOS draws from episodes where it almost never fires (base-SUCCESS offsets) and reaches
# -21.6% at p=0.63. Same target rule, same recorder, different amount of shield steering.
#
# THIS JOB IS THE MISSING CELL OF THAT COMPARISON. Same ten offsets as the round-1
# collection, same base policy, same recorder, same host, same target rule -- and the
# shield OFF. With no projection ever applied, executed action IS the clipped nominal, so
# no record transform is needed and none is applied: the difference from SFT0 is purely
# which states got visited.
#
#   NOSHIELD much weaker than SFT0  -> the state distribution is the active ingredient,
#                                      and the runtime shield earns its place as an
#                                      exploration device even though its corrections do not
#   NOSHIELD as good as SFT0        -> this is ordinary self-distillation, the shield
#                                      contributes nothing at all, and that is what gets
#                                      written
#
# The prediction is recorded here before the job runs, and the readout is the same one the
# positive result was measured with: paired Wilcoxon and sign test on policy-induced cost,
# success rate reported alongside as the guard, never netted into a single number.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

# exactly the offsets round-1 collection used
OFFS=(0 5 10 26 27 29 36 38 42 43)
OFF=${OFFS[$((SGE_TASK_ID - 1))]:-}
[ -n "$OFF" ] || { echo "no offset for task $SGE_TASK_ID"; exit 0; }

OUT=$LR/v38_noshield/off$OFF
RECROOT=$LR/v38_noshield/records
mkdir -p "$OUT" "$RECROOT" $LR/v38_noshield/batch
if [ -s "$OUT/result.json" ] && \
   external/VLA-Arena/envs/openpi/.venv/bin/python -c \
     'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="pass" else 1)' \
     "$OUT/result.json" 2>/dev/null; then
  echo "SKIP $OUT"; exit 0
fi
rm -f "$OUT/result.json"
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true
BLOBS_BEFORE=$( { find "$RECROOT/blobs" -type f 2>/dev/null || true; } | wc -l )

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
unset SE_VLA_POLICY_CHECKPOINT_DIR || true
# THE one variable: no shield. SRD stays enabled only so the recorder's plumbing runs;
# AEGIS_SOURCE=0 means no projection is ever computed or applied.
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_PHASE1_RECORD=1 SE_VLA_PHASE1_CANDIDATES=1
export SE_VLA_PHASE1_ROOT="$RECROOT"
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="nosh_off${OFF}"

echo "=== v38 shield-off collection off=$OFF on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((64000 + 20 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "V38_EXIT=$rc off=$OFF"

if [ ! -s "$OUT/result.json" ] || ! external/VLA-Arena/envs/openpi/.venv/bin/python -c \
     'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="pass" else 1)' \
     "$OUT/result.json" 2>/dev/null; then
  echo "V38_CELL_FAILED off=$OFF -- removing for retry"; rm -f "$OUT/result.json"; exit 1
fi

BLOBS_AFTER=$( { find "$RECROOT/blobs" -type f 2>/dev/null || true; } | wc -l )
# The whole point is that no projection ever fired. If one did, this is not the control it
# claims to be, and a shield-off arm that quietly had a shield would be the worst possible
# version of this experiment.
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$OFF" "$BLOBS_BEFORE" "$BLOBS_AFTER" <<'PYCHK'
import json, os, sys
out, off, before, after = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
trig = 0
p = os.path.join(out, "srd.jsonl")
if os.path.exists(p):
    for line in open(p):
        try: r = json.loads(line)
        except Exception: continue
        if r.get("type") == "aegis_projection" and r.get("triggered"):
            trig += 1
r = json.load(open(os.path.join(out, "result.json")))
ad, md = r.get("adapter", {}), r.get("metric_decomposition", {})
print(f"V38 off={off} aegis_triggers={trig} corrections_applied={ad.get('corrections_applied')} "
      f"sr={r.get('successes')} polcc={md.get('policy_induced_cc')} blobs {before}->{after}")
assert trig == 0, f"a projection fired {trig} times in the shield-OFF control"
assert ad.get("corrections_applied", 0) == 0, "a correction was applied in the shield-OFF control"
assert after > before, "the recorder wrote nothing"
print("V38_CELL_VERIFIED")
PYCHK
