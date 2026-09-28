#!/usr/bin/env bash
#$ -N SE_eval
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 4
#$ -t 1-94
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/se_r1/eval_batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/se_r1/eval_batch/j.$JOB_ID.$TASK_ID.out
#
# R1 evaluation -- the endpoint that matches the stated goal: crash-failure count.
#
# polCC ~= 270 x (episodes that fail INTO the hazard), so cost is a scaled harm count and
# the target is the crash count itself. D sits at 6 crash cells [0,1,12,26,29,42] and
# polCC 1714 on this stratum; the published shield reaches 1 but converts 11 successes into
# safe-but-failed episodes, a price the user explicitly does not want to pay. So BOTH
# directions are reported and a cost win bought with lost successes is a rejection.
#
# The shield runs OBSERVE-ONLY, exactly as in the C1 collection: the policy drives, the
# shield only computes and logs. That makes this a measurement of what the POLICY learned,
# not of what a runtime filter can rescue, and it is bit-comparable with the C1 baseline,
# which reproduced D's 41/47 / 1714 / [0,1,12,26,29,42] exactly.
#
# Budget 100 steps for all four arms: every arm passed the registered quiet-drift guard
# there (flow ratio 0.80-0.89 against the 1.10 limit), and it is the larger of the two
# budgets trained.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
SEC=$LR/se_r1
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python
STEPS=100

ARMS=(CURR STATIC)
OFFS=(0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47)
idx=$((SGE_TASK_ID - 1))
ai=$((idx / 47)); oi=$((idx % 47))
[ "$ai" -lt 2 ] || { echo "no cell for task $SGE_TASK_ID"; exit 0; }
ARM=${ARMS[$ai]}; OFF=${OFFS[$oi]}

SRC=$SEC/train/${ARM}
FOLD=$SEC/fold/${ARM}
OUT=$SEC/eval/${ARM}/off$OFF
mkdir -p "$OUT" "$SEC/eval_batch" "$FOLD"
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi

# ---- fold the adapter once per arm; concurrent cells race, so guard with a lock ----
CK=$FOLD/offset_0
if [ ! -d "$CK/params" ]; then
  LOCK=$FOLD/.foldlock
  if mkdir "$LOCK" 2>/dev/null; then
    PYTHONPATH=src:external/VLA-Arena:. $VENV - "$SRC" "$FOLD" <<'PYFOLD'
import pathlib, sys
OP = pathlib.Path("${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/models/openpi")
sys.path.insert(0, str(OP / "scripts"))
import fold_lora_adapter as F
import orbax.checkpoint as ocp
F.FOLDS = pathlib.Path(sys.argv[1]); F.OUT = pathlib.Path(sys.argv[2])
F.OUT.mkdir(parents=True, exist_ok=True)
F.fold_one(0, ocp.StandardCheckpointer(), verbose=True)
print("FOLD_OK")
PYFOLD
    rmdir "$LOCK"
  else
    for i in $(seq 1 120); do [ -d "$CK/params" ] && break; sleep 10; done
  fi
fi
[ -d "$CK/params" ] || { echo "SE1E_ABORT: no folded checkpoint at $CK"; exit 1; }
[ -e "$CK/assets" ] || ln -s ${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned/assets "$CK/assets" 2>/dev/null || true

hostname > "$OUT/host.txt"
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_POLICY_CHECKPOINT_DIR="$CK"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SHIELD_OBSERVE_ONLY=1
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="se1_${ARM}_off${OFF}"

echo "=== R1 eval arm=$ARM off=$OFF ckpt=$CK on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. $VENV \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((57000 + 20 * (SGE_TASK_ID % 200))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "SE1E_EXIT=$rc arm=$ARM off=$OFF"
[ -s "$OUT/result.json" ] || { echo "SE1E_CELL_FAILED arm=$ARM off=$OFF"; exit 1; }
$VENV - "$OUT" "$ARM" "$OFF" <<'PYV'
import json, sys, pathlib
out, arm, off = sys.argv[1], sys.argv[2], sys.argv[3]
r = json.load(open(pathlib.Path(out) / "result.json"))
md = r.get("metric_decomposition") or {}
print(f"SE1E_RESULT arm={arm} off={off} sr={r.get('successes')} polcc={md.get('policy_induced_cc')}")
assert pathlib.Path(out, "result.json").stat().st_size > 0
print("SE1E_CELL_VERIFIED")
PYV
