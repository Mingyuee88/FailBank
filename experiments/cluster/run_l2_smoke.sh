#!/usr/bin/env bash
#$ -N L2_smoke
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_D
#$ -pe smp 4
#$ -t 1-6
#$ -tc 3
#$ -o ${WORK_ROOT}/lora_dagger/l2_smoke/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/l2_smoke/batch/j.$JOB_ID.$TASK_ID.out
#
# FEASIBILITY SMOKE for safety_static_obstacles level_2.
#
# Why level_2 is the next target: its cost predicate names TWO hazards
# (red_coffee_mug_1 and red_coffee_mug_2, both Fall/InContact/CheckGripperContact), while
# AEGIS constrains exactly ONE object -- a single VLM-chosen name, a single point cloud, a
# single fitted ellipsoid (vlsa_port.perceive: name = obstacle_detection(...); p2,R2,Q2 =
# fit_ellipse(filt)). Either it protects one mug and leaves the other unguarded, or the
# enclosing ellipsoid swallows both plus the gap between them.
#
# Unlike the dynamic suite, level_2 needs no new geometry work: _crop_suite_for maps by
# SUITE, so the measured tabletop crop box already applies.
#
# This smoke only establishes that the pipeline runs at level 2 and that the shield engages;
# it is on a6k because 004/005 are saturated and a feasibility check needs no cross-host
# comparability. All comparison runs will be pinned to one L40S host.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/l2_smoke
i=$((SGE_TASK_ID - 1)); ARMI=$((i / 3)); OFF=$((i % 3))
case "$ARMI" in
  0) ARM=base   ;;
  1) ARM=aegis  ;;
  *) echo "no arm"; exit 0 ;;
esac
OUT=$OUTROOT/${ARM}/off$OFF
mkdir -p "$OUT" "$OUTROOT/batch"
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
unset SE_VLA_POLICY_CHECKPOINT_DIR || true
export SE_VLA_ARENA_SUITE=safety_static_obstacles
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
if [ "$ARM" = "aegis" ]; then
  : "${SE_VLA_GLM_BASE_URL:=http://HOST_E:8502/v1}"
  export SE_VLA_GLM_BASE_URL
  export SE_VLA_SHIELD_IMPL=vlsa_aegis SE_VLA_VLSA_DOF=3
  unset ZHIPUAI_API_KEY || true
else
  export SE_VLA_SHIELD_OBSERVE_ONLY=1
fi
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="l2_${ARM}_off${OFF}"
echo "=== L2 smoke arm=$ARM task=2(mango) level=2 off=$OFF on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 2 --task-id 2 --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((14000 + 7 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "L2_EXIT=$rc arm=$ARM off=$OFF"
[ -s "$OUT/result.json" ] || { echo "L2_FAILED arm=$ARM off=$OFF"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$ARM" "$OFF" <<'PYV'
import json, sys, pathlib
out, arm, off = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
r = json.load(open(out/"result.json")); md = r.get("metric_decomposition") or {}
sr = int((r.get("successes") or 0) > 0); cc = float(md.get("policy_induced_cc") or 0)
ti = r.get("task_identity") or {}
hazard = None; qp = None
srd = out/"srd.jsonl"
if srd.exists():
    for line in srd.open():
        try: x = json.loads(line)
        except Exception: continue
        if x.get("type") == "vlsa_audit":
            hazard = x.get("obstacle_name"); qp = x.get("qp_solved")
        if x.get("type") == "barrier_audit" and hazard is None:
            hazard = x.get("obstacle_count_sum")
print(f"L2_RESULT arm={arm} off={off} sr={sr} polcc={cc} bddl={ti.get('task_bddl_file')} "
      f"level={ti.get('task_level')} vlm_hazard={hazard!r} qp_solved={qp}")
assert "_2" in str(ti.get("task_bddl_file")), "level 2 bddl was not loaded"
print("L2_CELL_VERIFIED")
PYV
