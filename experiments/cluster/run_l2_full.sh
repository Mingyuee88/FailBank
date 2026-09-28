#!/usr/bin/env bash
#$ -N L2_full
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_D
#$ -pe smp 4
#$ -t 1-450
#$ -tc 3
#$ -o ${WORK_ROOT}/lora_dagger/l2_full/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/l2_full/batch/j.$JOB_ID.$TASK_ID.out
#
# LEVEL 2: the operating point where AEGIS's single-obstacle design must break.
#
# level_2's cost predicate names TWO hazards -- red_coffee_mug_1 and red_coffee_mug_2, each
# with Fall / InContact / CheckGripperContact. AEGIS constrains exactly one: a single
# VLM-chosen name, one point cloud, one fitted ellipsoid. The smoke confirmed the VLM returns
# the singular string 'red mug' while the shield genuinely runs (QP solved 145-300), so this
# is not a plumbing failure -- it recognises one obstacle out of two.
#
# Three arms, one host, three sampler-advance repeats each:
#   base   the policy alone (shield observe-only, no steering)
#   aegis  the published shield
#   ours   the curriculum policy pi_2, trained at real strength on L1t2 -- level_2 is unseen
#
# cost_pair_min_name is recorded per cell so the mechanism can be checked directly: if AEGIS
# only ever guards one mug, its residual cost should concentrate on the other one.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/l2_full
ARMS=(base aegis ours); ADV=(0 1 2)
i=$((SGE_TASK_ID - 1)); ai=$((i / 150)); rem=$((i % 150)); OFF=$((rem / 3)); vi=$((rem % 3))
[ "$ai" -lt 3 ] || { echo "no combo"; exit 0; }
ARM=${ARMS[$ai]}; A=${ADV[$vi]}
OUT=$OUTROOT/${ARM}/off${OFF}_r${A}
mkdir -p "$OUT" "$OUTROOT/batch"
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_ARENA_SUITE=safety_static_obstacles
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SAMPLER_ADVANCE=$A
unset SE_VLA_POLICY_CHECKPOINT_DIR || true
case "$ARM" in
  base)
    export SE_VLA_SHIELD_OBSERVE_ONLY=1 ;;
  aegis)
    : "${SE_VLA_GLM_BASE_URL:=http://HOST_E:8502/v1}"; export SE_VLA_GLM_BASE_URL
    export SE_VLA_SHIELD_IMPL=vlsa_aegis SE_VLA_VLSA_DOF=3
    unset ZHIPUAI_API_KEY || true ;;
  ours)
    export SE_VLA_SHIELD_OBSERVE_ONLY=1
    export SE_VLA_POLICY_CHECKPOINT_DIR=$LR/se_curr2/ckpt/CURRICULUM_p2/offset_0
    [ -d "$SE_VLA_POLICY_CHECKPOINT_DIR/params" ] || { echo "L2_ABORT: no pi_2"; exit 1; } ;;
esac
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="l2f_${ARM}_off${OFF}_r${A}"
echo "=== L2full arm=$ARM off=$OFF adv=$A on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 2 --task-id 2 --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((10000 + 3 * (SGE_TASK_ID % 1200))) --array-task-id 1 --server-attempts 3
set -e
[ -s "$OUT/result.json" ] || { echo "L2F_FAILED arm=$ARM off=$OFF adv=$A"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$ARM" "$OFF" "$A" <<'PYV'
import json, sys, pathlib, collections
out, arm, off, a = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
r = json.load(open(out/"result.json")); md = r.get("metric_decomposition") or {}
sr = int((r.get("successes") or 0) > 0); cc = float(md.get("policy_induced_cc") or 0)
guarded = None; pairs = collections.Counter()
srd = out/"srd.jsonl"
if srd.exists():
    for line in srd.open():
        try: x = json.loads(line)
        except Exception: continue
        if x.get("type") == "vlsa_audit": guarded = x.get("obstacle_name")
        pre = x.get("pre_step_features") or {}
        nm = pre.get("cost_pair_min_name")
        if nm: pairs[nm] += 1
top = pairs.most_common(2)
print(f"L2F_RESULT arm={arm} off={off} adv={a} sr={sr} polcc={cc} "
      f"guarded={guarded!r} nearest_pairs={top}")
PYV
