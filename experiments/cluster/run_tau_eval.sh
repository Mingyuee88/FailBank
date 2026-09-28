#!/usr/bin/env bash
#$ -N TauEval
#$ -cwd
#$ -q gpu@HOST_D
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-72
#$ -tc 4
#$ -j y
#
# Does hazard lookahead rescue the dynamic arm?
#
# Three arms on safety_dynamic_obstacles level_2 task 0, offsets 32-49 (both adapters trained
# on 0-31 only, so these 18 are unseen):
#   base   -- pi0.5-ft, no adapter
#   tau00  -- our recipe, hazard sphere at the hazard's CURRENT position (the original)
#   tau02  -- our recipe, hazard sphere advanced by v*0.2s (~1.12 oracle radii)
#
# tau00 is the control, not a second treatment: the only difference between the two adapters
# is which teacher labels the shield produced during collection. Both collections ran on
# a6k-002, both adapters trained with the same recipe (no staging, quiet_weight 0, 800 steps).
#
# Prior numbers to beat, from the earlier (mixed-host) round: base SR 88.9 / CC 3.17,
# tau=0 arm SR 83.3 / CC 9.39. Lookahead has to pull SR back above base AND push CC back
# toward it; moving only one of them is not a rescue.
#
# PINNED to a6k-002 -- same host as both collections. GPU model alone moves polCC 17.5%.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/tau_eval

i=$((SGE_TASK_ID - 1))
ARM_I=$((i / 18)); OFF=$((32 + i % 18))
case "$ARM_I" in
  0) ARM=base;  CK="" ;;
  1) ARM=tau00; CK=$LR/se_tau/TAU00_ho/ckpt/TAU00_HO/offset_0 ;;
  2) ARM=tau02q; CK=$LR/se_tau/TAU02q_ho/ckpt/TAU02Q_HO/offset_0 ;;
  3) ARM=tau01;  CK=$LR/se_tau/TAU01_ho/ckpt/TAU01_HO/offset_0 ;;
esac
CELL="${ARM}_off${OFF}"
OUT=$OUTROOT/$CELL
mkdir -p "$OUT"
if [ -s "$OUT/result.json" ]; then echo "SKIP $CELL"; exit 0; fi
hostname > "$OUT/host.txt"

if [ -n "$CK" ]; then
  [ -d "$CK/params" ] || { echo "TAUEVAL_ABORT: no checkpoint at $CK"; exit 1; }
  export SE_VLA_POLICY_CHECKPOINT_DIR="$CK"
else
  unset SE_VLA_POLICY_CHECKPOINT_DIR || true
fi

# Evaluation geometry is the ORIGINAL snapshot for every arm. Lookahead was a training-time
# choice about teacher labels; giving the tau02 arm a different shield at test time would
# confound "better policy" with "better shield" -- and the shield is observe-only here anyway.
export SE_VLA_HAZARD_LOOKAHEAD_S=0.0
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SHIELD_OBSERVE_ONLY=1
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="taueval_${CELL}"

for v in SE_VLA_SEED SE_VLA_RESULT_PATH SE_VLA_SRD_NEAR_DISTANCE_M \
         SE_VLA_SRD_RELEASE_DISTANCE_M SE_VLA_SRD_MIN_CLOSING_M_PER_STEP; do
  eval "test -n \"\${$v:-}\"" || { echo "MISSING_REQUIRED_ENV $v"; exit 1; }
done

echo "=== TauEval $CELL arm=$ARM ck=${CK:-<base>} eval_tau=$SE_VLA_HAZARD_LOOKAHEAD_S on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_dynamic_obstacles \
  --task-level 2 --task-id 0 --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((51000 + 20 * (SGE_TASK_ID - 1))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "TAUEVAL_EXIT=$rc cell=$CELL"
[ -s "$OUT/result.json" ] || { echo "TAUEVAL_CELL_FAILED $CELL"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/result.json" "$CELL" "$ARM" <<'PYCHK'
import json, sys
r = json.load(open(sys.argv[1])); md = r.get("metric_decomposition") or {}
print("TAUEVAL cell=%s arm=%s sr=%s official_cc=%s policy_cc=%s" % (
    sys.argv[2], sys.argv[3], r.get("successes"),
    md.get("official_cc"), md.get("policy_induced_cc")))
PYCHK
