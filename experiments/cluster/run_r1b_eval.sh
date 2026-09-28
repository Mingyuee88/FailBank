#!/usr/bin/env bash
#$ -N R1bEval
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 4
#$ -t 1-250
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/rounds2/batch/
#$ -j y
#
# Evaluate r1b -- round 1 retrained at round 2's budget.
#
# The original r1 trained 100 steps, r2 trained 400 (inconsistent script defaults), so the
# measured r1->r2 gain could not be separated from four times the training. r1b uses the
# SAME R1 records (se_curr/s1_records, 3657 rows, collected by pi0) at steps=400 batch=32.
# Its training metrics now match r2's almost exactly:
#     r1b  flow 1.02789  drift 0.005929
#     r2   flow 1.02524  drift 0.005917
# so r1b vs r2 differs in one thing only: which policy generated the training data.
#
# Same task (L1t2), same host, same output tree as the base/r1/r2 cells so it all pools.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/rounds2
ARM=r1b
i=$((SGE_TASK_ID - 1)); OFF=$((i / 5)); A=$((i % 5))
OUT=$OUTROOT/$ARM/off${OFF}_r${A}
mkdir -p "$OUT" "$OUTROOT/batch"
if [ -s "$OUT/result.json" ] && grep -q '"status": "pass"' "$OUT/result.json"; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_ARENA_SUITE=safety_static_obstacles
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0 SE_VLA_SHIELD_OBSERVE_ONLY=1
export SE_VLA_SAMPLER_ADVANCE=$A
export SE_VLA_POLICY_CHECKPOINT_DIR=$LR/se_curr1b/ckpt/CURRICULUM_p2/offset_0
[ -d "$SE_VLA_POLICY_CHECKPOINT_DIR/params" ] || { echo "RD_ABORT: no r1b checkpoint"; exit 1; }
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="rounds2_${ARM}_off${OFF}_r${A}"
echo "=== Rounds arm=$ARM off=$OFF adv=$A ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((24000 + 3 * (SGE_TASK_ID % 1200))) --array-task-id 1 --server-attempts 3
set -e
[ -s "$OUT/result.json" ] || { echo "RD_FAILED arm=$ARM off=$OFF adv=$A"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$ARM" "$OFF" "$A" <<'PYV'
import json, sys, pathlib
out, arm, off, a = (pathlib.Path(sys.argv[1]),) + tuple(sys.argv[2:5])
r = json.load(open(out/"result.json"))
if r.get("status") != "pass":
    print(f"RD_UNUSABLE arm={arm} off={off} adv={a} status={r.get('status')}"); sys.exit(0)
md = r.get("metric_decomposition") or {}
src = r.get("policy_source") or ""
print(f"RD_RESULT arm={arm} off={off} adv={a} sr={int((r.get('successes') or 0) > 0)} "
      f"polcc={float(md.get('policy_induced_cc') or 0)}")
PYV
