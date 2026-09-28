#!/usr/bin/env bash
#$ -N Pi0Rem
#$ -cwd
#$ -q gpu@HOST_A
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-200
#$ -tc 4
#$ -j y
#
# Cross-backbone evaluation: does the recipe transfer to pi0?
#
# Both arms are re-run here rather than reusing the existing pi0_base sweep, because that
# sweep was spread over l40s-004 and a6k-001 (37/13, 39/11, 48/2 per task). GPU model alone
# moves policy-induced cost by 17.5%, so a paired comparison against it would be invalid.
# EXPECT_HOST makes a stray cell delete its own result and fail rather than leave a
# plausible-looking but incomparable number.
#
# The pi0 adapter needed quiet_weight 0.5 to clear the acceptance guard (flow 1.083);
# lower settings and every step count were rejected. Its triggered loss is 0.178, far above
# the 0.003-0.03 seen on pi0.5 -- that gap is itself a finding about chunk length.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/pi0_eval2
EXPECT_HOST=HOST_A

i=$((SGE_TASK_ID - 1))
ARM_I=$((i / 100)); R=$((i % 100)); T=$((R / 50)); OFF=$((R % 50))
TIDS=(0 3); TID=${TIDS[$T]}
case "$ARM_I" in 0) ARM=base ;; 1) ARM=ours ;; esac
CELL="${ARM}_t${TID}_off${OFF}"
OUT=$OUTROOT/$CELL
mkdir -p "$OUT"
if [ -s "$OUT/result.json" ] && grep -q '"status": "pass"' "$OUT/result.json"; then echo "SKIP $CELL"; exit 0; fi
hostname > "$OUT/host.txt"
if [ "$(hostname -s)" != "$EXPECT_HOST" ]; then
  echo "HOST_MISMATCH expected=$EXPECT_HOST got=$(hostname -s) cell=$CELL"
  rm -f "$OUT/result.json"; exit 1
fi

export SE_VLA_OPENPI_YAML=${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/configs/evaluation/openpi_pi0.yaml
export SE_VLA_CONFIG_RESOLUTION=${SE_VLA_ROOT}/roundG/pi05_smoke/config_resolution_pi0.json
if [ "$ARM" = "ours" ]; then
  # Was se_pi0_q05/ckpt/... -- that checkpoint was folded onto the hardcoded pi05 base
  # in fold_lora_adapter.py and is pi05-structured (verify_family.py says family=pi05).
  # This one is the same adapter re-folded onto the pi0 base; verified family=pi0.
  CK=${WORK_ROOT}/lora_dagger/se_pi0_q05/ckpt_pi0fold_1405748/PI0_Q05/offset_0
  [ -d "$CK/params" ] || { echo "PI0EVAL_ABORT: no adapter at $CK"; exit 1; }
  export SE_VLA_POLICY_CHECKPOINT_DIR="$CK"
else
  unset SE_VLA_POLICY_CHECKPOINT_DIR || true
fi

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SHIELD_OBSERVE_ONLY=1
export SE_VLA_HAZARD_LOOKAHEAD_S=0.0
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="pi0ev_${CELL}"

for v in SE_VLA_SEED SE_VLA_RESULT_PATH SE_VLA_SRD_NEAR_DISTANCE_M \
         SE_VLA_SRD_RELEASE_DISTANCE_M SE_VLA_SRD_MIN_CLOSING_M_PER_STEP; do
  eval "test -n \"\${$v:-}\"" || { echo "MISSING_REQUIRED_ENV $v"; exit 1; }
done

echo "=== Pi0Ev $CELL arm=$ARM on $(hostname -s) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id "$TID" --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((56000 + 3 * (SGE_TASK_ID % 1200))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "PI0EV_EXIT=$rc cell=$CELL"
[ -s "$OUT/result.json" ] || { echo "PI0EV_CELL_FAILED $CELL"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/result.json" "$CELL" "$ARM" "$TID" "$OFF" <<'PYCHK'
import json, sys
r = json.load(open(sys.argv[1])); md = r.get("metric_decomposition") or {}
a = r.get("architecture") or {}
expected_ckpt = "${WORK_ROOT}/lora_dagger/se_pi0_q05/ckpt_pi0fold_1405748/PI0_Q05/offset_0"
assert r.get("status") == "pass", f"status={r.get('status')}"
assert a.get("architecture_family") == "pi0", f"resolved as {a.get('architecture_family')}"
assert r.get("task_suite_name") == "safety_static_obstacles"
assert r.get("task_level") == 1 and r.get("task_id") == int(sys.argv[4])
assert r.get("init_state_offset") == int(sys.argv[5])
if sys.argv[3] == "ours":
    assert a.get("adapter_checkpoint_dir") == expected_ckpt, a.get("adapter_checkpoint_dir")
else:
    assert not a.get("adapter_checkpoint_dir"), a.get("adapter_checkpoint_dir")
print("PI0EV cell=%s arm=%s sr=%s official_cc=%s policy_cc=%s" % (
    sys.argv[2], sys.argv[3], r.get("successes"),
    md.get("official_cc"), md.get("policy_induced_cc")))
PYCHK
