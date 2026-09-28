#!/usr/bin/env bash
#$ -N v32_eval
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -pe smp 2
#$ -t 1-150
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v32_eval/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v32_eval/batch/j.$JOB_ID.$TASK_ID.out
#
# v32_eval -- deployment evaluation for every arm in the attribution and round-2 sets.
#
# Every arm here deploys the same way arm A does: the folded weights are the entire
# method. Shield off, memory off, no wrapper, no runtime device. What differs between
# arms is only what was distilled into those weights.
#
#   SFT0    shield residual deleted            null for "generic fine-tuning helps"
#   SHAM    residual direction randomized,     null for "any matched perturbation helps"
#           magnitude preserved per record
#   SFTPOS  imitation of base's own successes  null for "ordinary BC matches distillation"
#   R2      round 1 + round 2 on R1's states   the self-evolving arm
#   R2CTRL  round 1 + round 2 on BASE's states matched-budget control for R2
#
# TWO DOMAINS, and only one of them decides anything:
#
#   dev   safety_static_obstacles L1t2, the 47 offsets everything was developed on.
#         Reported, but its held-out half consists only of offsets base already SUCCEEDS
#         on, so it can measure retention and regression-recovery and nothing else --
#         there is no headroom there to demonstrate repair. It is a probe, not a result.
#   xfer  L1t1/L1t3/L1t4, 150 offsets, lemon/onion/tomato. Nothing here took any part in
#         collection, training, or a single configuration choice, and the offsets were
#         NOT selected on base's outcome -- all 50 per task, failures and successes
#         alike. This is the endpoint. base=116/150, arm A=121/150 (p=0.533).
#
# READOUT fixed before submission: repair rate and retention rate reported separately and
# never netted (a net of zero can be "nothing happened" or "one repair traded for one new
# failure", which are different methods); per-task, never pooled into one binomial
# (pooling once hid a task being significantly harmed); exact McNemar paired on offsets;
# obj_travel checked in BOTH directions, since a policy that stops moving and one that
# flings the object both buy a beautiful cost number.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

: "${ARM:?ARM must be passed with qsub -v ARM=...}"
: "${DOMAIN:?DOMAIN must be passed with qsub -v DOMAIN=dev|xfer}"
CK=$LR/v31_fold/$ARM/offset_0
[ -d "$CK/params" ] || { echo "missing folded checkpoint $CK"; exit 1; }

# Domains. `dev` and `xfer` are the original two. `hazard` and `l2` are the powered
# endpoint the 750-cell census identified: four strata whose base SR is off both ceilings
# AND whose failures carry policy-induced cost, together 74 costly base-failure offsets
# against the 34 the original transfer set had. `hazard` matters most because its safety
# semantics are genuinely different -- a lit candle or a hot stove rather than a fruit
# beside a mug -- which is the "not the same template with a different object" objection.
# Each domain is pinned to the host its base census was measured on; that host is supplied
# on the qsub line and recorded per cell.
i=$((SGE_TASK_ID - 1))
SUITE=safety_static_obstacles
LEVEL=1
case "$DOMAIN" in
  dev)
    OFFS=(0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47)
    [ "$i" -lt "${#OFFS[@]}" ] || { echo "no cell for task $SGE_TASK_ID in dev"; exit 0; }
    TID=2; OFF=${OFFS[$i]} ;;
  xfer)
    TASKS=(1 3 4); TID=${TASKS[$((i / 50))]}; OFF=$((i % 50)) ;;
  hazard)
    SUITE=safety_hazard_avoidance
    TASKS=(2 3)
    [ "$i" -lt 100 ] || { echo "no cell for task $SGE_TASK_ID in hazard"; exit 0; }
    TID=${TASKS[$((i / 50))]}; OFF=$((i % 50)) ;;
  l2)
    LEVEL=2
    TASKS=(2 4)
    [ "$i" -lt 100 ] || { echo "no cell for task $SGE_TASK_ID in l2"; exit 0; }
    TID=${TASKS[$((i / 50))]}; OFF=$((i % 50)) ;;
  *) echo "unknown DOMAIN=$DOMAIN"; exit 1 ;;
esac

OUT=$LR/v32_eval/$ARM/$DOMAIN/L${LEVEL}t${TID}/off${OFF}
mkdir -p "$OUT" $LR/v32_eval/batch
# Resume rule: only a cell that PASSED counts as done. A transport crash mid-episode
# (the arena's websocket dying under load) writes a status=fail result.json with the
# partial products discarded; treating that as "done" would silently drop the cell from
# the arm forever and quietly bias the arm's denominator.
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
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="v32_${ARM}_${DOMAIN}_L${LEVEL}t${TID}_off${OFF}"
# run.py requires the three SRD distances even with the shield off
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
# the weights are the entire method
export SE_VLA_POLICY_CHECKPOINT_DIR="$CK"
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0

echo "=== v32 arm=$ARM domain=$DOMAIN $SUITE L${LEVEL}t${TID} off=$OFF ckpt=$CK on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite "$SUITE" \
  --task-level "$LEVEL" --task-id "$TID" --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((54000 + 20 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "V32_EXIT=$rc arm=$ARM domain=$DOMAIN L${LEVEL}t$TID off=$OFF"

# A failed episode must leave NO result.json behind, so the next submission retries it
# rather than inheriting a hole.
if [ ! -s "$OUT/result.json" ] || ! external/VLA-Arena/envs/openpi/.venv/bin/python -c \
     'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="pass" else 1)' \
     "$OUT/result.json" 2>/dev/null; then
  echo "V32_CELL_FAILED arm=$ARM domain=$DOMAIN L${LEVEL}t$TID off=$OFF -- removing partial result for retry"
  rm -f "$OUT/result.json"
  exit 1
fi

# The one failure mode that would corrupt every arm identically and invisibly: the
# checkpoint override not reaching the policy server, so all five "arms" are the base
# policy. Assert against the server's own restore log, not against the variable we set.
# Compare RESOLVED paths. Two arms address their checkpoint through a symlink -- `A`
# points at the older v20_fold tree and `SFTPOS`'s offset_0 points at its offset_1 fold --
# and orbax logs the path it actually opened, which is the resolved one. Grepping the
# unresolved path made this guard fire on 219 perfectly good cells, which is worse than
# no guard: a real mismatch would have looked exactly the same.
CK_REAL=$(readlink -f "$CK")
if ! grep -qE "Restoring checkpoint from ($CK|$CK_REAL)/params" "$OUT"/server_port*.log 2>/dev/null; then
  echo "CHECKPOINT_NOT_LOADED arm=$ARM off=$OFF"
  grep -i "Restoring checkpoint from" "$OUT"/server_port*.log 2>/dev/null | head -3 || true
  exit 1
fi
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/result.json" "$ARM" "$DOMAIN" "$TID" "$OFF" "$SUITE" "$LEVEL" <<'PYCHK'
import json, sys
r = json.load(open(sys.argv[1])); arm, dom, tid, off, suite, lvl = sys.argv[2:8]
md = r.get("metric_decomposition", {})
assert r.get("status") == "pass", f"episode status {r.get('status')}"
assert int(r.get("task_id", -1)) == int(tid) and int(r.get("init_state_offset", -1)) == int(off)
assert r.get("task_identity", {}).get("identity_gate") == "pass"
assert r.get("task_suite_name") == suite and int(r.get("task_level", -1)) == int(lvl)
assert r["adapter"].get("corrections_applied", 0) == 0, "a runtime correction fired in a weights-only arm"
print(f"V32_RESULT arm={arm} {dom} L{lvl}t{tid} off{off} sr={r.get('successes')} "
      f"polcc={md.get('policy_induced_cc')} officialcc={md.get('official_cc')}")
print("V32_CELL_VERIFIED")
PYCHK
