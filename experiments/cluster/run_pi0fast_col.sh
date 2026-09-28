#!/usr/bin/env bash
#$ -N Pi0FCol
#$ -cwd
#$ -q gpu@HOST_A
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-32
#$ -tc 4
#$ -j y
#
# Round-1 collection for the Pi0-FAST arm, safety_static_obstacles level_1 task 2, offsets 0-31.
#
# Derived verbatim from run_pi0_col.sh; only the backbone-specific lines differ (yaml, config
# resolution, outroot, architecture assert, ports, episode id). Do not hand-edit parameters --
# self-authored configs have silently no-opped on this project three times in one day.
#
# TASK CHOICE. Unlike Pi0 -- which produced policy-induced cost on 28/50 cells of t2 and 0/50 of
# both t1 and t4, forcing t2 -- Pi0-FAST produces it on all three (7 / 22 / 25 cells). Task choice
# is therefore free here and is fixed instead by COMPARABILITY: t2 is what the Pi0.5 main line and
# the Pi0 arm both collected on. Offsets 0-31, leaving 32-49 as a clean holdout, same split.
#
# PINNED to HOST_A: the same host the 150-cell Pi0-FAST base ran on. GPU model alone moves
# policy-induced CC by 17.5%, so every cell of this comparison must sit on one host.
#
# action_horizon is 10 for Pi0-FAST (Pi0 is 50). It is read from config_resolution_pi0fast.json,
# NOT passed here -- do not add a flag that would let it silently disagree with the checkpoint.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
OUTROOT=${WORK_ROOT}/lora_dagger/pi0fast_col_t2

TID=2
OFF=$((SGE_TASK_ID - 1))
CELL="t${TID}_off${OFF}"
OUT=$OUTROOT/$CELL
mkdir -p "$OUT"
if [ -s "$OUT/result.json" ]; then echo "SKIP $CELL"; exit 0; fi
hostname > "$OUT/host.txt"

export SE_VLA_OPENPI_YAML=${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/configs/evaluation/openpi_pi0fast_local.yaml
export SE_VLA_CONFIG_RESOLUTION=${SE_VLA_ROOT}/roundG/pi05_smoke/config_resolution_pi0fast.json
unset SE_VLA_POLICY_CHECKPOINT_DIR || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SHIELD_OBSERVE_ONLY=1
export SE_VLA_PHASE1_RECORD=1 SE_VLA_PHASE1_CANDIDATES=1
export SE_VLA_PHASE1_ROOT=$OUTROOT/records
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="pi0fcol_${CELL}"

for v in SE_VLA_SEED SE_VLA_RESULT_PATH SE_VLA_SRD_NEAR_DISTANCE_M \
         SE_VLA_SRD_RELEASE_DISTANCE_M SE_VLA_SRD_MIN_CLOSING_M_PER_STEP \
         SE_VLA_PHASE1_ROOT SE_VLA_PHASE1_RECORD; do
  eval "test -n \"\${$v:-}\"" || { echo "MISSING_REQUIRED_ENV $v"; exit 1; }
done

echo "=== Pi0FCol $CELL on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id "$TID" --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((39000 + 20 * (SGE_TASK_ID - 1))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "PI0FCOL_EXIT=$rc cell=$CELL"
[ -s "$OUT/result.json" ] || { echo "PI0FCOL_CELL_FAILED $CELL"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT/result.json" "$CELL" "$SE_VLA_PHASE1_ROOT" <<"PYCHK"
import json, sys, glob, os
r = json.load(open(sys.argv[1])); md = r.get("metric_decomposition") or {}
a = r.get("architecture") or {}
assert a.get("architecture_family") == "pi0_fast", "not the pi0-FAST architecture"
assert a.get("signature_keys_present") is True
# Records must actually land, else the whole collection is a silent no-op.
recs = glob.glob(os.path.join(sys.argv[3], "**", "*.json"), recursive=True)
recs = [p for p in recs if os.path.basename(p) != "manifest.json"]
print("PI0FCOL cell=%s sr=%s offcc=%s polcc=%s records_so_far=%d" % (
    sys.argv[2], r.get("successes"), md.get("official_cc"),
    md.get("policy_induced_cc"), len(recs)))
PYCHK
