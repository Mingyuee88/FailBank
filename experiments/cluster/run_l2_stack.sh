#!/usr/bin/env bash
#$ -N L2stack
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_D
#$ -pe smp 4
#$ -t 1-150
#$ -tc 3
#$ -o ${WORK_ROOT}/lora_dagger/l2_stack/batch/
#$ -j y
#
# pi_2 WITH the AEGIS shield actually in the loop -- a combination that has never been run.
# Every previous "ours" arm used SE_VLA_SHIELD_OBSERVE_ONLY=1, i.e. the shield computed and
# logged but the environment stepped the policy's own action. (The known "shield made things
# worse" episode was about DATA: an in-loop shield repaired the crashes we were trying to
# collect, leaving the failure bank empty. That is a collection artefact, not a performance
# result, and it says nothing about deployment.)
#
# Why it should help, from the level_2 mango decomposition (150 cells/arm):
#     bucket                ours          aegis
#     grazing success   105 @ 11.9     24 @  5.5
#     crash               9 @ 172.4    10 @ 154.1
#     clean success          36           108
#     safe-fail               0             8
# The crash column is already at parity; the whole polCC gap (1118 of 1129 points) is the
# clean-success column. AEGIS buys those 108 clean cells with a projection step that keeps the
# eef out of the unsafe set -- exactly the thing a learned policy has no hard mechanism for.
# Conversely pi_2 supplies the SR (47.0 vs 44.0) and never abandons the task (0 vs 8 cells).
# The hypothesis is that the two are complementary rather than redundant.
#
# Same host as l2_full so the arms are directly comparable (GPU model alone moves polCC 17.5%).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/l2_stack
CK=$LR/se_curr2/ckpt/CURRICULUM_p2/offset_0
[ -d "$CK/params" ] || { echo "L2S_ABORT: no pi_2 at $CK"; exit 1; }
i=$((SGE_TASK_ID - 1)); OFF=$((i / 3)); A=$((i % 3))
OUT=$OUTROOT/ours_aegis/off${OFF}_r${A}
mkdir -p "$OUT" "$OUTROOT/batch"
if [ -s "$OUT/result.json" ] && grep -q '"status": "pass"' "$OUT/result.json"; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_ARENA_SUITE=safety_static_obstacles
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SAMPLER_ADVANCE=$A
# the stack: curriculum policy + published shield IN THE LOOP (no observe-only)
export SE_VLA_POLICY_CHECKPOINT_DIR="$CK"
: "${SE_VLA_GLM_BASE_URL:=http://HOST_E:8502/v1}"; export SE_VLA_GLM_BASE_URL
export SE_VLA_SHIELD_IMPL=vlsa_aegis SE_VLA_VLSA_DOF=3
unset ZHIPUAI_API_KEY || true
unset SE_VLA_SHIELD_OBSERVE_ONLY || true
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="l2stack_off${OFF}_r${A}"
echo "=== L2stack ours+aegis off=$OFF adv=$A ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 2 --task-id 2 --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((12000 + 3 * (SGE_TASK_ID % 1200))) --array-task-id 1 --server-attempts 3
set -e
[ -s "$OUT/result.json" ] || { echo "L2S_FAILED off=$OFF adv=$A"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$OFF" "$A" <<'PYV'
import json, sys, pathlib
out, off, a = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
r = json.load(open(out/"result.json"))
if r.get("status") != "pass":
    print(f"L2S_UNUSABLE off={off} adv={a} status={r.get('status')}"); sys.exit(0)
md = r.get("metric_decomposition") or {}
# assert the shield genuinely ran: an unset suite or a dead VLM yields sr/polcc that read
# like a clean result. This exact failure produced a fake "AEGIS fails" once already.
guarded = None; qp = 0
srd = out/"srd.jsonl"
if srd.exists():
    for line in srd.open(errors="ignore"):
        try: x = json.loads(line)
        except Exception: continue
        if x.get("type") == "vlsa_audit" and x.get("obstacle_name"): guarded = x["obstacle_name"]
        if x.get("type") == "aegis_projection": qp += 1
ck = (r.get("policy_source") or "") 
print(f"L2S_RESULT off={off} adv={a} sr={int((r.get('successes') or 0) > 0)} "
      f"polcc={float(md.get('policy_induced_cc') or 0)} guarded={guarded!r} qp_rows={qp} "
      f"shield_ran={guarded is not None}")
PYV
