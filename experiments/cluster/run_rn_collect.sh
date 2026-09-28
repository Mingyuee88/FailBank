#!/usr/bin/env bash
#$ -N RnCol
#$ -cwd
#$ -q gpu@HOST_A
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-47
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/rn_collect_batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/rn_collect_batch/j.$JOB_ID.$TASK_ID.out
#
# C1 -- collect the failure signal the bank does not have.
#
# WHY. Every one of the 2830 existing learning records has `eventual_success = True`: the
# outcome filter kept only successful episodes, and only 20 episodes were ever collected,
# over 10 offsets, all under BASE. Per-step cost labels are almost absent (H1 any_cost
# 16/2830 = 0.57%). So the policy has never seen a crash in training, which is the most
# likely reason its crash count stalls at 6 while the published shield reaches 1 -- and
# crash count is what polCC measures, since polCC ~= 270 x crash-failure count.
#
# WHAT CHANGES vs run_phase1_collect.sh:
#   1. the policy is D, not base -- self-evolution collects on the CURRENT policy's own
#      state distribution, which is the only distribution its next round will act in;
#   2. all 47 dev offsets, not 10. Collecting only on cells that crash would bias the
#      state distribution toward failure and, given the forgetting measured on 2026-08-19
#      (support regression 18/20 on held-out records the policy was not trained on),
#      would put the 41 cells it already succeeds on at risk. The composition rule is
#      therefore "every cell, unfiltered";
#   3. NO outcome filter. Failures are the point. Filtering happens later, at staging.
#
# THE SHIELD RUNS IN OBSERVE-ONLY MODE. This is the correction to the first attempt.
#
# With the shield IN THE LOOP, the trajectory never approaches the hazard at all: on D's own
# failure cells off0 and off1 the shielded run reached cost_pair_min_distance 0.183 / 0.175
# -- far above the 0.1087 risk threshold -- with zero contacts and sr=1, polCC=0. The
# rescued run contains no crash-bound state, which is precisely what this stage exists to
# collect. Evidence kept at c1_collect_shieldinloop/.
#
# `SE_VLA_SHIELD_OBSERVE_ONLY=1` makes the shield compute its correction and record it as
# the learning target while the environment steps the POLICY'S OWN action. That is the
# DAgger form -- query the expert on the STUDENT'S state distribution -- and it gives both
# halves at once: trajectories that actually go wrong, and a target at every step of them.
# Turning the shield off entirely would give the trajectories but no target, and negative
# learning on a flow-matching policy is unvalidated here.
#
# Shield parameters are copied verbatim from manifest_01.json of the original collection so
# the new records are commensurable with the old ones.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=${SE_COLLECT_OUT:?SE_COLLECT_OUT must be set}
CK=${SE_COLLECT_CK:?SE_COLLECT_CK must be set (policy used to collect)}
[ -d "$CK" ] || { echo "C1_ABORT: missing checkpoint $CK"; exit 1; }

OFFS=(0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47)
i=$((SGE_TASK_ID - 1))
[ "$i" -lt "${#OFFS[@]}" ] || { echo "no cell for task $SGE_TASK_ID"; exit 0; }
OFF=${OFFS[$i]}

OUT=$OUTROOT/off$OFF
mkdir -p "$OUT" "$OUTROOT/batch"
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_POLICY_CHECKPOINT_DIR="$CK"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SHIELD_OBSERVE_ONLY=1
export SE_VLA_PHASE1_RECORD=1 SE_VLA_PHASE1_CANDIDATES=1
export SE_VLA_PHASE1_ROOT=$OUTROOT/records
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="c1_D_off${OFF}"

echo "=== C1 collect off=$OFF policy=D on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((55000 + 20 * (SGE_TASK_ID - 1))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "C1_EXIT=$rc off=$OFF"
[ -s "$OUT/result.json" ] || { echo "C1_CELL_FAILED off=$OFF"; exit 1; }

# Effective-value check on the run.py patch: the cost-pair snapshot must actually be in the
# records. If it is silently empty the whole staging axis is unbuildable, and this is exactly
# the class of failure AGENTS.md documents four times over -- artifacts that look healthy.
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUTROOT/records" "$OFF" "$OUT" <<'PYCHK'
import json, pathlib, sys
root, off, out = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
res = json.load(open(out / "result.json"))
md = res.get("metric_decomposition") or {}
print(f"C1_RESULT off={off} sr={res.get('successes')} polcc={md.get('policy_induced_cc')}")
# Locate THIS cell's episode by identity, never by mtime: all 47 cells share one records
# root and run concurrently, so "the newest file" is very often another cell's episode.
# (Observed 2026-08-19: off0 and off1 both validated against the same episode and printed
# identical min-distance, which read as a broken --offset until the records were opened.)
want = f"c1_D_off{off}"
newest = None
for meta in root.glob("episodes/*/*/episode.json"):
    try:
        if json.load(meta.open()).get("task_description") == want:
            newest = meta.parent / "raw_steps.jsonl"
            break
    except Exception:
        continue
assert newest is not None and newest.exists(), f"no episode recorded for {want}"
n = have = contacts = 0
dmin = None
for line in newest.open():
    r = json.loads(line)
    ri = r.get("runtime_info") or {}
    n += 1
    d = ri.get("cost_pair_min_distance")
    if d is not None:
        have += 1
        dmin = d if dmin is None else min(dmin, d)
    contacts += int(ri.get("cost_pair_contacts") or 0)
print(f"  episode={newest.parent.name[:12]} steps={n} with_cost_pair={have} "
      f"min_distance={dmin} contact_steps={contacts}")
assert have > 0, "cost_pair_min_distance absent from every record -- the run.py patch is a no-op"
assert have == n, f"cost_pair present on only {have}/{n} steps"
assert dmin is not None and dmin > 0, f"cost_pair_min_distance is degenerate ({dmin})"
# The observe-only switch is the load-bearing one: if it silently did nothing, the shield
# would be steering again and every trajectory would be a rescued one, giving a bank with
# no failures in it -- healthy-looking and useless.
flags = set()
for line in newest.open():
    flags.add(bool((json.loads(line).get("runtime_info") or {}).get("shield_observe_only")))
assert flags == {True}, f"shield_observe_only not set on every step: {flags}"
print("  observe_only=True on all steps")
print("C1_CELL_VERIFIED")
PYCHK
