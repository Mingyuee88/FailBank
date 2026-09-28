#!/usr/bin/env bash
#$ -N DynEval
#$ -cwd
#$ -q gpu@HOST_D
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-108
#$ -tc 4
#$ -j y
#
# Two-level generalisation test for the dynamic arm, both levels against the same base.
#
#   level 1 (unseen initial states): safety_dynamic_obstacles level_2 task 0, offsets 32-49.
#       The adapter trained on offsets 0-31 ONLY (se_dyn_L2t0_ho), so these 18 are genuinely
#       unseen. The full-offset arm is deliberately not evaluated here -- it saw 0..47 and
#       would only measure memorisation.
#   level 2 (unseen dynamic tasks): level_1 tasks 3 and 4, offsets 0-17. Chosen from the
#       64-cell scan: t3 has headroom (SR 37.5) and t4 has the thickest policy cost (4/8).
#       t1/t2 and L2t2 sit at SR 100 and would show nothing either way.
#
# PINNED to one host on purpose: GPU model alone moves policy-induced CC by 17.5%, and both
# arms of a comparison must sit on the same silicon. Every cell here is a6k-002.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/dyn_eval
HO_CK=$LR/se_dyn_L2t0_ho/ckpt/DYN_L2t0_HO/offset_0

i=$((SGE_TASK_ID - 1))
ARM_I=$((i / 54)); R=$((i % 54)); G=$((R / 18)); K=$((R % 18))
case "$ARM_I" in
  0) ARM=base ;;
  1) ARM=ho ;;
esac
case "$G" in
  0) LV=2; TID=0; OFF=$((32 + K)); LEVEL_NAME=offset ;;   # unseen initial states
  1) LV=1; TID=3; OFF=$K;          LEVEL_NAME=task ;;     # unseen task
  2) LV=1; TID=4; OFF=$K;          LEVEL_NAME=task ;;     # unseen task
esac
CELL="${ARM}_L${LV}t${TID}_off${OFF}"
OUT=$OUTROOT/$CELL
mkdir -p "$OUT"
if [ -s "$OUT/result.json" ]; then echo "SKIP $CELL"; exit 0; fi
hostname > "$OUT/host.txt"

if [ "$ARM" = "ho" ]; then
  [ -d "$HO_CK/params" ] || { echo "DYNEVAL_ABORT: no holdout checkpoint at $HO_CK"; exit 1; }
  export SE_VLA_POLICY_CHECKPOINT_DIR="$HO_CK"
else
  unset SE_VLA_POLICY_CHECKPOINT_DIR || true
fi

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SHIELD_OBSERVE_ONLY=1     # no safety layer: all seven forms failed, pi_2 runs bare
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="dyneval_${CELL}"

for v in SE_VLA_SEED SE_VLA_RESULT_PATH SE_VLA_SRD_NEAR_DISTANCE_M \
         SE_VLA_SRD_RELEASE_DISTANCE_M SE_VLA_SRD_MIN_CLOSING_M_PER_STEP; do
  eval "test -n \"\${$v:-}\"" || { echo "MISSING_REQUIRED_ENV $v"; exit 1; }
done

echo "=== DynEval $CELL arm=$ARM level=$LEVEL_NAME on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_dynamic_obstacles \
  --task-level "$LV" --task-id "$TID" --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((47000 + 20 * (SGE_TASK_ID - 1))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "DYNEVAL_EXIT=$rc cell=$CELL"
[ -s "$OUT/result.json" ] || { echo "DYNEVAL_CELL_FAILED $CELL"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/result.json" "$CELL" "$ARM" "$LEVEL_NAME" <<'PYCHK'
import json, sys
r = json.load(open(sys.argv[1])); md = r.get("metric_decomposition") or {}
print("DYNEVAL cell=%s arm=%s level=%s sr=%s offcc=%s polcc=%s" % (
    sys.argv[2], sys.argv[3], sys.argv[4], r.get("successes"),
    md.get("official_cc"), md.get("policy_induced_cc")))
PYCHK
