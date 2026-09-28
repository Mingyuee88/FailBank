#!/usr/bin/env bash
#$ -N TauEvT4
#$ -cwd
#$ -q gpu@HOST_D
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-144
#$ -tc 4
#$ -j y
#
# Evaluate on safety_dynamic_obstacles level_1 task 4, offsets 0-17. UNSEEN TASK: every
# adapter here was trained on L2 t0 only.
#
# Why move off L2t0's holdout: on offsets 32-49 the base policy scores clean on 16 of 18
# cells, so at most 2 cells can possibly change. The paired sign test over those 18 came back
# p=1.000 for every arm pair -- not because the arms are equal, but because the design cannot
# resolve anything. L1t3's base sits at SR 61.1 with 7 safe-fail cells, giving real headroom.
#
# Three arms, same host as every other measurement in this comparison:
#   base    -- pi0.5-ft
#   tau00   -- recipe with the original snapshot geometry
#   tau02q  -- lookahead 0.2s + quiet_weight 0.2
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/tau_eval_t4

i=$((SGE_TASK_ID - 1))
ARM_I=$((i / 18)); OFF=$((i % 18))
case "$ARM_I" in
  0) ARM=base;   CK="" ;;
  1) ARM=tau00;  CK=$LR/se_tau/TAU00_ho/ckpt/TAU00_HO/offset_0 ;;
  2) ARM=tau02q; CK=$LR/se_tau/TAU02q_ho/ckpt/TAU02Q_HO/offset_0 ;;
  3) ARM=bankq01; CK=$LR/se_bank_r2_ho/ckpt/BANK800Q01/offset_0 ;;
  4) ARM=bankq02; CK=$LR/se_bank_r2_ho/ckpt/BANK800Q02/offset_0 ;;
  5) ARM=bank700; CK=$LR/se_bank_r2_ho/ckpt/BANK700/offset_0 ;;
  6) ARM=bank3; CK=$LR/se_bank_r3b_ho/ckpt/BANK3_700/offset_0 ;;
  7) ARM=bank3e; CK=$LR/se_bank_r3b_ho/ckpt/BANK3_1150/offset_0 ;;
esac
CELL="${ARM}_off${OFF}"
OUT=$OUTROOT/$CELL
mkdir -p "$OUT"
if [ -s "$OUT/result.json" ]; then echo "SKIP $CELL"; exit 0; fi
hostname > "$OUT/host.txt"

if [ -n "$CK" ]; then
  [ -d "$CK/params" ] || { echo "T4EVAL_ABORT: no checkpoint at $CK"; exit 1; }
  export SE_VLA_POLICY_CHECKPOINT_DIR="$CK"
else
  unset SE_VLA_POLICY_CHECKPOINT_DIR || true
fi

# Original snapshot geometry for every arm at test time: lookahead was a training-time choice
# about teacher labels, and the shield is observe-only here anyway.
export SE_VLA_HAZARD_LOOKAHEAD_S=0.0
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SHIELD_OBSERVE_ONLY=1
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="t4eval_${CELL}"

for v in SE_VLA_SEED SE_VLA_RESULT_PATH SE_VLA_SRD_NEAR_DISTANCE_M \
         SE_VLA_SRD_RELEASE_DISTANCE_M SE_VLA_SRD_MIN_CLOSING_M_PER_STEP; do
  eval "test -n \"\${$v:-}\"" || { echo "MISSING_REQUIRED_ENV $v"; exit 1; }
done

echo "=== TauEvalT3 $CELL arm=$ARM (unseen task L1t3) on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_dynamic_obstacles \
  --task-level 1 --task-id 4 --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((55000 + 20 * (SGE_TASK_ID - 1))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "T4EVAL_EXIT=$rc cell=$CELL"
[ -s "$OUT/result.json" ] || { echo "T4EVAL_CELL_FAILED $CELL"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/result.json" "$CELL" "$ARM" <<'PYCHK'
import json, sys
r = json.load(open(sys.argv[1])); md = r.get("metric_decomposition") or {}
print("T4EVAL cell=%s arm=%s sr=%s official_cc=%s policy_cc=%s" % (
    sys.argv[2], sys.argv[3], r.get("successes"),
    md.get("official_cc"), md.get("policy_induced_cc")))
PYCHK
