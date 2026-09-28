#!/usr/bin/env bash
#$ -N Rounds2
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 4
#$ -t 1-750
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/rounds2/batch/
#$ -j y
#
# THE ACTUAL multi-round comparison.
#
# The earlier `rounds` run compared se_curr2's phase1 against its phase2 -- but
# run_curriculum2.sh trains BOTH phases on se_r2 records, with phase1 starting from
# ORIG_BASE. That is batching one dataset, which is the same variable the CURRICULUM /
# STATIC arms already isolated. It says nothing whatsoever about rounds, and the
# "multi-round has no effect" conclusion drawn from it is withdrawn.
#
# The real self-evolution chain:
#     pi0 --R1 records--> se_curr/CURRICULUM_p2                     (round 1)
#         that model is then RUN to collect fresh failures
#         (run_r2_collect.sh: CK=se_curr/ckpt/CURRICULUM_p2)
#     --se_r2 records--> se_curr2/CURRICULUM_p2                     (round 2)
#
# Round 2 trains on what round 1's own policy failed at. That is the claim being tested.
#     base  pi0, no failure-bank training
#     r1    se_curr/CURRICULUM_p2   trained on R1 records
#     r2    se_curr2/CURRICULUM_p2  trained on R2 records, which r1 generated
#
# Same task (L1t2) and host (HOST_A) as the earlier evaluations so everything pools.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/rounds2
ARMS=(base r1 r2)
i=$((SGE_TASK_ID - 1)); ai=$((i / 250)); rem=$((i % 250)); OFF=$((rem / 5)); A=$((rem % 5))
ARM=${ARMS[$ai]}
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
case "$ARM" in
  base) unset SE_VLA_POLICY_CHECKPOINT_DIR || true ;;
  r1)   export SE_VLA_POLICY_CHECKPOINT_DIR=$LR/se_curr/ckpt/CURRICULUM_p2/offset_0 ;;
  r2)   export SE_VLA_POLICY_CHECKPOINT_DIR=$LR/se_curr2/ckpt/CURRICULUM_p2/offset_0 ;;
esac
if [ "$ARM" != "base" ]; then
  [ -d "$SE_VLA_POLICY_CHECKPOINT_DIR/params" ] || { echo "RD_ABORT: no $ARM checkpoint"; exit 1; }
fi
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="rounds2_${ARM}_off${OFF}_r${A}"
echo "=== Rounds arm=$ARM off=$OFF adv=$A ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((22000 + 3 * (SGE_TASK_ID % 1200))) --array-task-id 1 --server-attempts 3
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
