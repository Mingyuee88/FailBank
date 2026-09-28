#!/usr/bin/env bash
#$ -N CU_eval
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 4
#$ -t 1-94
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/se_curr/eval_batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/se_curr/eval_batch/j.$JOB_ID.$TASK_ID.out
#
# Evaluate the two-phase curriculum arms.
#
#   CURRICULUM  phase1 = s1 (S1_early only)   -> fold -> phase2 = s1s2
#   STATIC      phase1 = s1s2                 -> fold -> phase2 = s1s2
#
# Identical two-phase structure, one fold each, 100+100 steps, identical phase-2 data and
# starting checkpoint. The only variable is whether phase 1 withheld S2_mid, i.e. release
# ORDER -- the thing "easy to hard" actually claims and the one variable no earlier arm in
# this project ever changed.
#
# Checkpoints are already folded by run_curriculum.sh, so this script only evaluates.
# Shield runs OBSERVE-ONLY: the policy drives and the shield only logs, so this measures
# what the POLICY learned, and is bit-comparable with the C1 baseline that reproduced D's
# 41/47 / polCC 1714 / crash cells [0,1,12,26,29,42] exactly.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
SEC=$LR/se_curr
VENV=external/VLA-Arena/envs/openpi/.venv/bin/python

ARMS=(CURRICULUM STATIC)
OFFS=(0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47)
idx=$((SGE_TASK_ID - 1)); ai=$((idx / 47)); oi=$((idx % 47))
[ "$ai" -lt 2 ] || { echo "no cell for task $SGE_TASK_ID"; exit 0; }
ARM=${ARMS[$ai]}; OFF=${OFFS[$oi]}

CK=$SEC/ckpt/${ARM}_p2/offset_0
[ -d "$CK/params" ] || { echo "CUE_ABORT: no checkpoint at $CK"; exit 1; }
OUT=$SEC/eval/${ARM}/off$OFF
mkdir -p "$OUT" "$SEC/eval_batch"
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_POLICY_CHECKPOINT_DIR="$CK"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SHIELD_OBSERVE_ONLY=1
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="cu_${ARM}_off${OFF}"

echo "=== CU eval arm=$ARM off=$OFF ckpt=$CK on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. $VENV \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((58000 + 20 * (SGE_TASK_ID % 200))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "CUE_EXIT=$rc arm=$ARM off=$OFF"
[ -s "$OUT/result.json" ] || { echo "CUE_CELL_FAILED arm=$ARM off=$OFF"; exit 1; }
$VENV - "$OUT" "$ARM" "$OFF" <<'PYV'
import json, sys, pathlib
out, arm, off = sys.argv[1], sys.argv[2], sys.argv[3]
r = json.load(open(pathlib.Path(out) / "result.json"))
md = r.get("metric_decomposition") or {}
print(f"CUE_RESULT arm={arm} off={off} sr={r.get('successes')} polcc={md.get('policy_induced_cc')}")
print("CUE_CELL_VERIFIED")
PYV
