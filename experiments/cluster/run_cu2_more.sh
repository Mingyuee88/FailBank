#!/usr/bin/env bash
#$ -N Cu2More
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 4
#$ -t 1-470
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/cu2_eval/batch/
#$ -j y
#
# More repeats for the ONE comparison that isolates staged release.
#
# Existing 47 offsets x 5 repeats, clustered by offset (repeats of one scene are not
# independent trials):
#     CURRICULUM  SR 43.4  crash 6.6  polCC 1879  |  STATIC  SR 40.4  crash 9.6  polCC 2533
#     per-offset wins 14 / losses 5 / ties 28 -> sign test p=0.064 (not significant)
#     bootstrap dSR +0.06 [+0.01,+0.11] and dpolCC -13.1 [-24.7,-1.9] (both exclude 0)
# Direction is consistent across SR, crash and polCC, but the effect is small and 28 of 47
# offsets tie outright. That is a power problem, not a null result -- so add repeats.
#
# Repeats 5..9 (SE_VLA_SAMPLER_ADVANCE), same host as the existing five so the pooled set
# stays single-machine. Cells already present are skipped.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/cu2_eval
ARMS=(CURRICULUM STATIC)
i=$((SGE_TASK_ID - 1)); ai=$((i / 235)); rem=$((i % 235)); OFF=$((rem / 5)); vi=$((rem % 5))
ARM=${ARMS[$ai]}; A=$((vi + 5))          # repeats 5..9, the existing set is 0..4/6
CK=$LR/se_curr2/ckpt/${ARM}_p2/offset_0
[ -d "$CK/params" ] || { echo "CU2_ABORT: no $ARM at $CK"; exit 1; }
OUT=$OUTROOT/$ARM/off${OFF}_r${A}
mkdir -p "$OUT" "$OUTROOT/batch"
if [ -s "$OUT/result.json" ] && grep -q '"status": "pass"' "$OUT/result.json"; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_POLICY_CHECKPOINT_DIR="$CK"
export SE_VLA_ARENA_SUITE=safety_static_obstacles
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0 SE_VLA_SHIELD_OBSERVE_ONLY=1
export SE_VLA_SAMPLER_ADVANCE=$A
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="cu2_${ARM}_off${OFF}_r${A}"
echo "=== Cu2More arm=$ARM off=$OFF adv=$A ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((19000 + 3 * (SGE_TASK_ID % 1200))) --array-task-id 1 --server-attempts 3
set -e
[ -s "$OUT/result.json" ] || { echo "CU2_FAILED arm=$ARM off=$OFF adv=$A"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$ARM" "$OFF" "$A" <<'PYV'
import json, sys, pathlib
out, arm, off, a = (pathlib.Path(sys.argv[1]),) + tuple(sys.argv[2:5])
r = json.load(open(out/"result.json"))
if r.get("status") != "pass":
    print(f"CU2_UNUSABLE arm={arm} off={off} adv={a} status={r.get('status')}"); sys.exit(0)
md = r.get("metric_decomposition") or {}
print(f"CU2_RESULT arm={arm} off={off} adv={a} sr={int((r.get('successes') or 0) > 0)} "
      f"polcc={float(md.get('policy_induced_cc') or 0)}")
PYV
