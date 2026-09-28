#!/usr/bin/env bash
#$ -N v13_xfer
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_B
#$ -pe smp 2
#$ -t 1-300
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/v13_xfer/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/v13_xfer/batch/j.$JOB_ID.$TASK_ID.out
#
# v13_xfer -- the powered, zero-shot, cross-task test.
#
# The memory was harvested on L1t2 (mango). It is deployed here, unchanged, on
# L1t1 / L1t3 / L1t4 (lemon / onion / tomato). Every one of the 34 base-failure
# offsets is held out by construction: none was ever used for harvesting, for
# probing, or for choosing any configuration. That removes at a stroke the two
# things that limited every earlier result:
#
#   POWER    L1t2 offered 6 held-out failures -> 95% CI on a fix rate about
#            [3%, 56%]. Here there are 34 -> worst-case half-width about 16
#            points. Still short of the ~39 an 80%-power test would want, but a
#            different regime.
#   LEAKAGE  no offset here has been touched by any earlier run, so nothing needs
#            to be split into "training" and "held out" after the fact.
#
# It is also the generalisation test in the literal sense: one memory, three
# manipulated objects it has never seen.
#
# Base reference: v11_merged, same 150 offsets, same host, same run.py. The
# pipeline is bit-deterministic once the host is pinned, so the paired difference
# carries no noise term.
#
# ARMS. Two configurations earned a place, for opposite reasons, and neither is a
# tuned point -- both come from mechanism, not from a sweep maximum:
#
#   sched6  6-key retrieval + gap-scheduled magnitude. On L1t2 held-out: fix 1/6,
#           retention 35/36, dCC -22.3%. Best RETENTION of any arm that moved CC.
#   geo5    5-key retrieval (episode_step_index dropped) + the same schedule, no
#           abstain. On L1t2 held-out: fix 4/6, retention 31/36, dCC -13.0%. Best
#           FIX RATE of any memory arm, approaching the shield's 4/5 ceiling, but
#           it fires on 100% of steps and pays 5 base successes for it.
#
# The two rates are reported separately and never netted: geo5 and sched6 have
# nearly the same net dSR and completely different behaviour.
#
# Preregistered before submission: the primary readouts are (a) held-out fix rate
# with its Wilson interval, (b) retention rate with its interval, (c) paired dCC,
# (d) obj_travel ratio as the degenerate-solution guard, all on all 34 failures
# and all 116 successes. No offset is excluded after seeing results.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger

ARMS=(sched6 geo5)
TASKS=(1 3 4)
NOFF=50
i=$((SGE_TASK_ID - 1))
ARM=${ARMS[$((i / 150))]}
j=$((i % 150))
TID=${TASKS[$((j / NOFF))]}
OFF=$((j % NOFF))

OUT=$LR/v13_xfer/$ARM/L1t${TID}/off${OFF}
mkdir -p "$OUT" $LR/v13_xfer/batch
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="xfer_${ARM}_L1t${TID}_off${OFF}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_RELATION_MEMORY_AUTHORITY_GAIN=1.0
export SE_VLA_RELATION_MEMORY_MAGNITUDE_MODE=schedule
unset SE_VLA_RELATION_MEMORY_ABSTAIN_DISTANCE || true
unset SE_VLA_RELATION_MEMORY_RISK_GATE || true
unset SE_VLA_RELATION_MEMORY_MAGNITUDE_COEFFS || true

if [ "$ARM" = sched6 ]; then
  export SE_VLA_RELATION_MEMORY_FEATURE_KEYS="cost_pair_min_distance,cost_pair_delta_xy[0],cost_pair_delta_xy[1],eef_hazard_distance,eef_target_distance,episode_step_index"
  WANT=6
else
  export SE_VLA_RELATION_MEMORY_FEATURE_KEYS="cost_pair_min_distance,cost_pair_delta_xy[0],cost_pair_delta_xy[1],eef_hazard_distance,eef_target_distance"
  WANT=5
fi

CK=$(ls -d $LR/v7_l1t2/newkey-h5/round_*/settled_checkpoint.json 2>/dev/null | tail -1)
[ -n "$CK" ] || { echo "no settled checkpoint"; exit 1; }

# RUNTIME SELF-PROOF -- four silent config no-ops have already cost whole arms in
# this project, the last one 47 cells of a key ablation that ran with the
# checkpoint's stored key list. Prove the override lands before spending the GPU.
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python - "$CK" "$WANT" <<'PYCHK'
import json, os, sys
sys.path.insert(0, "src")
from se_vla.adapter.relation_memory import RelationOverrideMemory
d = json.load(open(sys.argv[1])); want = int(sys.argv[2])
mem = d if "entries" in d else next(v for v in d.values() if isinstance(v, dict) and "entries" in v)
m = RelationOverrideMemory.from_dict(mem)
m.feature_keys = [k for k in os.environ["SE_VLA_RELATION_MEMORY_FEATURE_KEYS"].split(",") if k]
assert len(m.feature_keys) == want, f"expected {want} keys, got {m.feature_keys}"
if want == 5:
    assert "episode_step_index" not in m.feature_keys, m.feature_keys
fit = m.fit_magnitude_schedule()
assert fit is not None and fit[1] < 0, f"schedule fit failed: {fit}"
print("XFER_CONFIG_VERIFIED keys", len(m.feature_keys), "slope", round(fit[1], 4))
PYCHK

echo "=== v13 arm=$ARM L1t${TID} off=$OFF ckpt=$CK on $(hostname) ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id "$TID" --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm pruned_memory --checkpoint "$CK" \
  --port-base $((2000 + 20 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
echo "XFER_EXIT=$? arm=$ARM L1t$TID off=$OFF"
