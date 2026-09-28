#!/usr/bin/env bash
#$ -N HOeval
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_C
#$ -pe smp 4
#$ -t 1-250
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/nc_eval/batch/
#$ -j y
#
# Evaluate the no-curriculum / quiet-weight checkpoints.
#
# Training outcome (800 steps, single phase, full record set):
#     R2_q00  accepted  drift 0.009492      no staging, quiet weight 0
#     R2_q20  accepted  drift 0.005139      no staging, quiet weight 0.2
#     R3_q00  REFUSED by the quiet-drift guard
#     R3_q20  accepted  drift 0.008514      round 3, only possible with quiet weight
#
# Two things this decides:
#   1. R2_q00 vs R2_q20 -- does weighting the successful (quiet) steps help? Those are
#      56-58% of the records and carried weight 0 in every earlier round, so they never
#      contributed. They are also the replay anchor against forgetting the previous round.
#   2. R2_q20 vs R3_q20 -- the multi-round claim, now with the successes included. Round 3
#      has only 1459 triggered records; without the quiet half it cannot even be trained
#      (R3_q00 above), which is itself the answer to "can self-evolution continue once
#      failures dry up".
#
# base is re-run here so all four arms share one host and one task (L1t2).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/ho_eval
# ARM is parameterised so the matching base arm can be submitted with -v ARM=base.
# Default unchanged: a bare submit still evaluates HOLDOUT exactly as before.
# The base arm is REQUIRED for the cross-offset claim: the existing HOLDOUT 250 cells
# are on HOST_C while rounds2/base is on HOST_A, and GPU model alone moves
# SR by up to 29.3 points -- larger than the 25.5-point effect being claimed.
ARM=${ARM:-HOLDOUT}
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
case "$ARM" in
  base) unset SE_VLA_POLICY_CHECKPOINT_DIR || true ;;
  *)    export SE_VLA_POLICY_CHECKPOINT_DIR=$LR/nocurr/ckpt/${ARM}/offset_0 ;;
esac
if [ "$ARM" != "base" ]; then
  [ -d "$SE_VLA_POLICY_CHECKPOINT_DIR/params" ] || { echo "NCE_ABORT: no $ARM checkpoint"; exit 1; }
fi
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="rounds2_${ARM}_off${OFF}_r${A}"
echo "=== Rounds arm=$ARM off=$OFF adv=$A ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((42000 + 3 * (SGE_TASK_ID % 1200))) --array-task-id 1 --server-attempts 3
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
