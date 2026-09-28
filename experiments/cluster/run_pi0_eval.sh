#!/usr/bin/env bash
#$ -N Pi0Eval
#$ -cwd
#$ -q gpu@HOST_C
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-108
#$ -tc 4
#$ -j y
#
# Does the recipe transfer to a DIFFERENT base model?
#
# Two arms, both Pi0: the Arena-official Pi0 finetune, and that same checkpoint after our
# single-round failure-bank training (no curriculum, quiet_weight 0, 800 steps) -- the exact
# recipe that worked on Pi0.5.
#
#   level 1 (unseen initial states): static L1 t2, offsets 32-49. Collection used 0-31 only.
#   level 2 (unseen tasks):          static L1 t1 and t4, offsets 0-17.
#
# Read the two levels differently. On t2 Pi0 carries policy cost on 28/50 cells, so both SR
# and CC are meaningful. On t1/t4 Pi0's policy-induced cost is 0/50 -- there is no safety
# signal to improve, so those two tasks test only whether the adapter preserves (or damages)
# task competence, where the base sits at SR 66.0 and 56.0.
#
# CC is reported under BOTH definitions. official_cc is what the Arena leaderboard reports;
# policy_induced_cc subtracts cost attributable to the initial state. They differ by 10x on
# Pi0 (8.01 vs 0.81 over 150 baseline cells), so a bare "CC" number is meaningless here.
#
# PINNED to a6k-001: GPU model alone moves policy CC by 17.5%.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/pi0_eval
TR_CK=$LR/se_pi0_t2/ckpt/PI0_T2/offset_0

i=$((SGE_TASK_ID - 1))
ARM_I=$((i / 54)); R=$((i % 54)); G=$((R / 18)); K=$((R % 18))
case "$ARM_I" in 0) ARM=base ;; 1) ARM=trained ;; esac
case "$G" in
  0) TID=2; OFF=$((32 + K)); LEVEL_NAME=offset ;;
  1) TID=1; OFF=$K;          LEVEL_NAME=task ;;
  2) TID=4; OFF=$K;          LEVEL_NAME=task ;;
esac
CELL="${ARM}_t${TID}_off${OFF}"
OUT=$OUTROOT/$CELL
mkdir -p "$OUT"
if [ -s "$OUT/result.json" ]; then echo "SKIP $CELL"; exit 0; fi
hostname > "$OUT/host.txt"

# Both arms resolve against the Pi0 config pair; the trained arm only swaps the checkpoint.
export SE_VLA_OPENPI_YAML=${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/configs/evaluation/openpi_pi0.yaml
export SE_VLA_CONFIG_RESOLUTION=${SE_VLA_ROOT}/roundG/pi05_smoke/config_resolution_pi0.json
if [ "$ARM" = "trained" ]; then
  [ -d "$TR_CK/params" ] || { echo "PI0EVAL_ABORT: no trained checkpoint at $TR_CK"; exit 1; }
  export SE_VLA_POLICY_CHECKPOINT_DIR="$TR_CK"
else
  unset SE_VLA_POLICY_CHECKPOINT_DIR || true
fi

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SHIELD_OBSERVE_ONLY=1     # no safety layer; all seven forms failed on Pi0.5
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="pi0eval_${CELL}"

for v in SE_VLA_SEED SE_VLA_RESULT_PATH SE_VLA_SRD_NEAR_DISTANCE_M \
         SE_VLA_SRD_RELEASE_DISTANCE_M SE_VLA_SRD_MIN_CLOSING_M_PER_STEP; do
  eval "test -n \"\${$v:-}\"" || { echo "MISSING_REQUIRED_ENV $v"; exit 1; }
done

echo "=== Pi0Eval $CELL arm=$ARM level=$LEVEL_NAME on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id "$TID" --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((50000 + 20 * (SGE_TASK_ID - 1))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "PI0EVAL_EXIT=$rc cell=$CELL"
[ -s "$OUT/result.json" ] || { echo "PI0EVAL_CELL_FAILED $CELL"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/result.json" "$CELL" "$ARM" "$LEVEL_NAME" <<'PYCHK'
import json, sys
r = json.load(open(sys.argv[1])); md = r.get("metric_decomposition") or {}
a = r.get("architecture") or {}
# the trained arm is a LoRA fold of a Pi0 checkpoint -- it must still resolve as Pi0
assert a.get("architecture_family") == "pi0", f"arm resolved as {a.get('architecture_family')}"
assert a.get("signature_keys_present") is True
print("PI0EVAL cell=%s arm=%s level=%s sr=%s official_cc=%s policy_cc=%s" % (
    sys.argv[2], sys.argv[3], sys.argv[4], r.get("successes"),
    md.get("official_cc"), md.get("policy_induced_cc")))
PYCHK
