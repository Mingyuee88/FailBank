#!/usr/bin/env bash
#$ -N L2task
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-900
#$ -tc 4
#$ -j y
#
# LEVEL 2, parameterised by task. Generalises run_l2_full.sh (which was hard-wired to the
# mango task) so the single-obstacle hypothesis can be tested across level_2.
#
# The benchmark supplies a clean controlled variable: every level_1 task names ONE hazard,
# every level_2 task names TWO. On level_1 AEGIS beat us on crash in all three tasks by
# +3.8/+6.9/+4.2 (noise bands +-0.6/+-2.3/+-2.7). On level_2 mango the gap vanished
# (crash 3.0 vs 3.3, band +-1.3) while the VLM named the singular 'red mug' in all 150
# cells -- one obstacle recognised out of two.
#
# Two tasks are run here to separate the two candidate explanations:
#   onion (TID=3)  wine_bottle_1 + wine_bottle_2      -- same-kind pair, VLM likely conflates
#   apple (TID=0)  white_yellow_mug_1 + wine_bottle_1 -- DIFFERENT kinds, so a VLM that can
#                  distinguish them still has to pick one. If AEGIS degrades here too, the
#                  defect is single-selection itself, not name ambiguity.
#
# TID and the host are passed with `qsub -v TID=.. -l h=..`; the host is pinned because GPU
# model alone shifts polCC by 17.5% and can flip SR (see memory: cc-seed-nondeterminism).
set -euo pipefail
: "${TID:?set TID via qsub -v TID=<task id>}"
: "${TAG:?set TAG via qsub -v TAG=<name>}"
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/l2_$TAG
# MISNOMER WARNING: the arm literally named "ours" below resolves to se_curr2/CURRICULUM_p2,
# i.e. the CURRICULUM checkpoint -- NOT the paper's "ours". Per the project naming rule,
# "ours" always means nocurr, which here is ncs1/ncs2/ncs3 (nocurr_s1/s2/s3, R2_q00).
# When tabulating, read the checkpoint path, never the arm name. The name is kept only
# because it is embedded in ~3000 already-completed cell directory names; renaming it
# would orphan that data and defeat the SKIP logic. Relabel at analysis time instead.
ARMS=(base aegis ours ncs1 ncs2 ncs3); ADV=(0 1 2)
i=$((SGE_TASK_ID - 1)); ai=$((i / 150)); rem=$((i % 150)); OFF=$((rem / 3)); vi=$((rem % 3))
[ "$ai" -lt "${#ARMS[@]}" ] || { echo "no combo"; exit 0; }
ARM=${ARMS[$ai]}; A=${ADV[$vi]}
OUT=$OUTROOT/${ARM}/off${OFF}_r${A}
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
unset SE_VLA_POLICY_CHECKPOINT_DIR || true
case "$ARM" in
  base)  export SE_VLA_SHIELD_OBSERVE_ONLY=1 ;;
  aegis) : "${SE_VLA_GLM_BASE_URL:=http://HOST_E:8502/v1}"; export SE_VLA_GLM_BASE_URL
         export SE_VLA_SHIELD_IMPL=vlsa_aegis SE_VLA_VLSA_DOF=3
         unset ZHIPUAI_API_KEY || true ;;
  ours)  export SE_VLA_SHIELD_OBSERVE_ONLY=1
         export SE_VLA_POLICY_CHECKPOINT_DIR=$LR/se_curr2/ckpt/CURRICULUM_p2/offset_0 ;;
  ncs1)  export SE_VLA_SHIELD_OBSERVE_ONLY=1
         export SE_VLA_POLICY_CHECKPOINT_DIR=$LR/nocurr_s1/ckpt/R2_q00/offset_0 ;;
  ncs2)  export SE_VLA_SHIELD_OBSERVE_ONLY=1
         export SE_VLA_POLICY_CHECKPOINT_DIR=$LR/nocurr_s2/ckpt/R2_q00/offset_0 ;;
  ncs3)  export SE_VLA_SHIELD_OBSERVE_ONLY=1
         export SE_VLA_POLICY_CHECKPOINT_DIR=$LR/nocurr_s3/ckpt/R2_q00/offset_0 ;;
esac
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="l2${TAG}_${ARM}_off${OFF}_r${A}"
echo "=== L2 $TAG (tid=$TID) arm=$ARM off=$OFF adv=$A ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 2 --task-id "$TID" --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((10000 + 3 * (SGE_TASK_ID % 1200))) --array-task-id 1 --server-attempts 3
set -e
[ -s "$OUT/result.json" ] || { echo "L2T_FAILED tag=$TAG arm=$ARM off=$OFF adv=$A"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$TAG" "$ARM" "$OFF" "$A" <<'PYV'
import json, sys, pathlib, collections
out, tag, arm, off, a = (pathlib.Path(sys.argv[1]),) + tuple(sys.argv[2:6])
r = json.load(open(out/"result.json"))
if r.get("status") != "pass":
    print(f"L2T_UNUSABLE tag={tag} arm={arm} off={off} adv={a} status={r.get('status')}")
    sys.exit(0)
md = r.get("metric_decomposition") or {}
guarded = None; pairs = collections.Counter()
srd = out/"srd.jsonl"
if srd.exists():
    for line in srd.open():
        try: x = json.loads(line)
        except Exception: continue
        if x.get("type") == "vlsa_audit": guarded = x.get("obstacle_name")
        nm = (x.get("pre_step_features") or {}).get("cost_pair_min_name")
        if nm: pairs[nm] += 1
print(f"L2T_RESULT tag={tag} arm={arm} off={off} adv={a} "
      f"sr={int((r.get('successes') or 0) > 0)} polcc={float(md.get('policy_induced_cc') or 0)} "
      f"guarded={guarded!r} nearest={pairs.most_common(2)}")
PYV
