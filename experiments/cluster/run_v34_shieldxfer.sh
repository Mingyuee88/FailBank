#!/usr/bin/env bash
#$ -N v34_shx
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -pe smp 2
#$ -t 1-100
#$ -tc 2
#$ -o ${WORK_ROOT}/lora_dagger/v34_shieldxfer/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v34_shieldxfer/batch/j.$JOB_ID.$TASK_ID.out
#
# v34_shieldxfer -- does the misalignment diagnosis, and its fix, survive a different
# safety semantics?
#
# Everything about the barrier/cost misalignment so far is one arena family: a fruit
# beside a mug, where the cost predicate is
#     fall(mug) AND incontact(fruit, mug) AND checkgrippercontact(mug).
# On that family the gripper-centred shield destroys 9 of 36 base successes, and those
# nine are exactly the offsets where base's own policy-induced cost was <= 7 while every
# repaired offset had >= 225 (AUC 0.000, p=0.00024). Re-centring the barrier on the
# manipulated object recovers 7 of the 9, and running both barriers in sequence is
# better than the original on every axis at once: repair 11/11 vs 10/11, retention 29/36
# vs 27/36, net 40/47 vs 37/47, cost still down 82%.
#
# All of that could still be an artefact of one cost predicate. `safety_hazard_avoidance`
# is the test: the hazard is a LIT CANDLE or a HOT STOVE, not a mug that can be knocked
# over, and the 750-cell base census puts L1t2 and L1t3 in the usable band (base 33/50
# and 10/50, and every one of their 57 failures carries policy-induced cost, median 284
# and 440). `safety_static_obstacles` L2 is the second domain: same semantics as the
# development arena but a harder level, which separates "different cost predicate" from
# "harder instance".
#
# TWO ARMS ONLY. `eef` is the published shield and the reference; `dual` is the fix. The
# `object`-only arm is omitted here -- it was informative as a mechanism probe on the
# development arena, where it showed the destroyed offsets coming back, but it gives up
# half the cost reduction and is not a candidate teacher.
#
# PREDICTION, fixed before the run. If the diagnosis generalises: on these domains too,
# `eef` destroys base successes concentrated at low base cost, and `dual` retains more of
# them without losing repairs. If `dual` is not better here, the fix is specific to the
# mug predicate and will be reported as such.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

: "${ARM:?ARM must be passed with qsub -v ARM=eef|dual}"
: "${DOMAIN:?DOMAIN must be passed with qsub -v DOMAIN=hazard|l2}"
i=$((SGE_TASK_ID - 1))
case "$DOMAIN" in
  hazard) SUITE=safety_hazard_avoidance; LEVEL=1; TASKS=(2 3) ;;
  l2)     SUITE=safety_static_obstacles; LEVEL=2; TASKS=(2 4) ;;
  *) echo "unknown DOMAIN=$DOMAIN"; exit 1 ;;
esac
[ "$i" -lt 100 ] || { echo "no cell for task $SGE_TASK_ID"; exit 0; }
TID=${TASKS[$((i / 50))]}
OFF=$((i % 50))

OUT=$LR/v34_shieldxfer/$ARM/$DOMAIN/L${LEVEL}t${TID}/off${OFF}
mkdir -p "$OUT" $LR/v34_shieldxfer/batch
if [ -s "$OUT/result.json" ] && \
   external/VLA-Arena/envs/openpi/.venv/bin/python -c \
     'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="pass" else 1)' \
     "$OUT/result.json" 2>/dev/null; then
  echo "SKIP $OUT"; exit 0
fi
rm -f "$OUT/result.json"
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
# base policy under the shield, identical recipe to every earlier shield measurement
unset SE_VLA_POLICY_CHECKPOINT_DIR || true
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_AEGIS_BARRIER_CENTER="$ARM"
export SE_VLA_AEGIS_GRASP_GATE_M=0.12
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="v34_${ARM}_${DOMAIN}_L${LEVEL}t${TID}_off${OFF}"

echo "=== v34 shield=$ARM $SUITE L${LEVEL}t${TID} off=$OFF on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite "$SUITE" \
  --task-level "$LEVEL" --task-id "$TID" --offset "$OFF" --seed 17 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((58000 + 20 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "V34_EXIT=$rc arm=$ARM domain=$DOMAIN L${LEVEL}t$TID off=$OFF"

if [ ! -s "$OUT/result.json" ] || ! external/VLA-Arena/envs/openpi/.venv/bin/python -c \
     'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="pass" else 1)' \
     "$OUT/result.json" 2>/dev/null; then
  echo "V34_CELL_FAILED arm=$ARM domain=$DOMAIN L${LEVEL}t$TID off=$OFF -- removing for retry"
  rm -f "$OUT/result.json"
  exit 1
fi

# Which barrier actually fired, counted by the code that fired it -- never the variable
# this script set itself.
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$ARM" "$DOMAIN" "$TID" "$OFF" "$SUITE" "$LEVEL" <<'PYCHK'
import json, os, sys
out, arm, dom, tid, off, suite, lvl = sys.argv[1:8]
audit = None
for line in open(os.path.join(out, "srd.jsonl")):
    try: r = json.loads(line)
    except Exception: continue
    if r.get("type") == "barrier_audit":
        audit = r
assert audit is not None, "no barrier_audit row -- the patched exit flush did not run"
assert audit["barrier_center"] == arm, f"audit says {audit['barrier_center']}, arm is {arm}"
if arm == "eef":
    assert audit["obj_fire"] == 0, "eef arm must never run the object barrier"
else:
    # The object barrier needs cost_pair_target_position. On a suite whose cost predicate
    # the feature extractor does not parse, that field is empty, the object term is
    # skipped on every step, and `dual` silently runs as plain `eef` -- producing a null
    # that means "the knob did nothing", not "the fix does not generalise". Fail loudly
    # instead of reporting that as a result.
    steps = max(int(audit["steps"]), 1)
    assert audit["obj_unavailable"] / steps < 0.5, (
        f"object barrier had no object on {audit['obj_unavailable']}/{steps} steps -- "
        f"this arm is INAPPLICABLE on this suite, not null")
r = json.load(open(os.path.join(out, "result.json")))
assert r.get("task_suite_name") == suite and int(r.get("task_level", -1)) == int(lvl)
assert int(r.get("task_id", -1)) == int(tid) and int(r.get("init_state_offset", -1)) == int(off)
md = r.get("metric_decomposition", {})
print(f"V34_RESULT arm={arm} {dom} L{lvl}t{tid} off{off} sr={r.get('successes')} "
      f"polcc={md.get('policy_induced_cc')} eef_fire={audit['eef_fire']} "
      f"obj_fire={audit['obj_fire']} obj_gated_out={audit['obj_gated_out']}")
print("V34_CELL_VERIFIED")
PYCHK
