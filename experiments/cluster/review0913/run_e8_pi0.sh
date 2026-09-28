#!/usr/bin/env bash
#$ -N RvE8Ev
#$ -cwd
#$ -q gpu@HOST_A
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-500
#$ -tc 4
#$ -j y
#
# Review 2026-09-13, E8 evaluation: pi0 adapters trained with lambda_q=0 and the first k
# actions supervised (k=10, 50). Both passed the guard (flow 1.017 / 1.000), so by the
# pre-registered rule they are evaluated on pi0 L1 T0-T4, 50 offsets, one condition.
#
# Everything below the arm/task mapping is copied from run_pi0_eval2.sh, so each cell pairs
# with the pi0_eval2 base arm (250/250 pass, all on HOST_A). Only the output root,
# arm names, checkpoint paths and episode-id prefix differ; a provenance check is added.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
O=$LR/review0913
OUTROOT=$O/e8/pi0_l1
EXPECT_HOST=HOST_A

i=$((SGE_TASK_ID - 1))
ARM_I=$((i / 250)); R=$((i % 250)); T=$((R / 50)); OFF=$((R % 50))
TIDS=(0 1 2 3 4); TID=${TIDS[$T]}
case "$ARM_I" in 0) ARM=pi0_k10 ;; 1) ARM=pi0_k50 ;; esac
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
# Folded onto the pi0 base by review0913/fold.sh; verify_family.py reported family=pi0.
CK=$O/e8/ckpt/$ARM/offset_0
[ -d "$CK/params" ] || { echo "PI0EVAL_ABORT: no adapter at $CK"; exit 1; }
export SE_VLA_POLICY_CHECKPOINT_DIR="$CK"

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SHIELD_OBSERVE_ONLY=1
export SE_VLA_HAZARD_LOOKAHEAD_S=0.0
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="rve8_${CELL}"

for v in SE_VLA_SEED SE_VLA_RESULT_PATH SE_VLA_SRD_NEAR_DISTANCE_M \
         SE_VLA_SRD_RELEASE_DISTANCE_M SE_VLA_SRD_MIN_CLOSING_M_PER_STEP; do
  eval "test -n \"\${$v:-}\"" || { echo "MISSING_REQUIRED_ENV $v"; exit 1; }
done

echo "=== RvE8Ev $CELL arm=$ARM on $(hostname -s) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id "$TID" --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((56000 + 3 * (SGE_TASK_ID % 1200))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "PI0EV_EXIT=$rc cell=$CELL"
[ -s "$OUT/result.json" ] || { echo "PI0EV_CELL_FAILED $CELL"; exit 1; }
if ! grep -ahq "Finished restoring checkpoint.* from $CK" "$OUT"/server_port*.log; then
  echo "PROVENANCE_MISMATCH cell=$CELL expected=$CK"
  mv "$OUT/result.json" "$OUT/result.provenance_failed.json"; exit 1
fi
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/result.json" "$CELL" "$ARM" <<'PYCHK'
import json, sys
r = json.load(open(sys.argv[1])); md = r.get("metric_decomposition") or {}
a = r.get("architecture") or {}
assert a.get("architecture_family") == "pi0", f"resolved as {a.get('architecture_family')}"
print("PI0EV cell=%s arm=%s sr=%s official_cc=%s policy_cc=%s" % (
    sys.argv[2], sys.argv[3], r.get("successes"),
    md.get("official_cc"), md.get("policy_induced_cc")))
PYCHK
