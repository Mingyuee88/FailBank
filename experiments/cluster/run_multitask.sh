#!/usr/bin/env bash
#$ -N MultiT
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-1000
#$ -tc 4
#$ -j y
#
# Multi-task validation, reported in the VLSA/AEGIS metric system.
#
# That paper (arXiv 2512.11891, Table 1 "Quantitative results on the SafeLIBERO benchmark")
# reports three metrics and does NOT report cumulative cost at all:
#     CAR (up)  collision avoidance rate -- % of episodes with a strictly collision-free
#               trajectory. With polCC == contact-steps (verified per cell), CAR is exactly
#               the share of cells whose polCC is 0, i.e. clean + safe-fail.
#     TSR (up)  task success rate -- % completed within the horizon.
#     ETS (down) execution time steps, averaged over all episodes including timeouts.
#
# Two things this run must deliver:
#   1. d021's advantage is so far only measured on L1t3. Three more tasks decide whether it
#      generalises -- that is the thinnest part of the current claim.
#   2. r1b -> r2 is directionally positive but not significant on one task
#      (13 wins / 8 losses, dSR +0.032, CI [-0.024,+0.088]). Pooling across tasks raises the
#      power roughly as sqrt(n_tasks), which is the only lever left: repeats are already 5
#      per offset and all 50 offsets are in use.
#
# Arms: base(pi0) / r1b(one round) / r2(two rounds) / d021(r2 + pre-contact retreat) / aegis
# Host is pinned per task; each task's five arms share one machine.
set -euo pipefail
: "${TID:?}"; : "${TAG:?}"
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=$LR/multi_$TAG
ARMS=(base r1b r2 d021 aegis nocurr ncs1 ncs2 ncs3 nc_shield)
i=$((SGE_TASK_ID - 1)); ai=$((i / 100)); rem=$((i % 100)); OFF=$((rem / 2)); A=$((rem % 2))
ARM=${ARMS[$ai]}
OUT=$OUTROOT/$ARM/off${OFF}_r${A}
mkdir -p "$OUT" "$OUTROOT/batch"
if [ -s "$OUT/result.json" ] && grep -q '"status": "pass"' "$OUT/result.json"; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
if [ -n "${EXPECT_HOST:-}" ] && [ "$(hostname -s)" != "$EXPECT_HOST" ]; then
  echo "HOST_MISMATCH expected=$EXPECT_HOST got=$(hostname -s) cell=$OUT"
  rm -f "$OUT/result.json"
  exit 1
fi
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_ARENA_SUITE=safety_static_obstacles
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SAMPLER_ADVANCE=$A
case "$ARM" in
  base)  export SE_VLA_SHIELD_OBSERVE_ONLY=1; unset SE_VLA_POLICY_CHECKPOINT_DIR || true ;;
  r1b)   export SE_VLA_SHIELD_OBSERVE_ONLY=1 SE_VLA_POLICY_CHECKPOINT_DIR=$LR/se_curr1b/ckpt/CURRICULUM_p2/offset_0 ;;
  r2)    export SE_VLA_SHIELD_OBSERVE_ONLY=1 SE_VLA_POLICY_CHECKPOINT_DIR=$LR/se_curr2/ckpt/CURRICULUM_p2/offset_0 ;;
  d021)  export SE_VLA_SHIELD_OBSERVE_ONLY=1 SE_VLA_POLICY_CHECKPOINT_DIR=$LR/se_curr2/ckpt/CURRICULUM_p2/offset_0
         export SE_VLA_ABORT_CONTACT_N=1 SE_VLA_ABORT_DISTANCE_M=0.21 \
                SE_VLA_RETREAT_MODE=auto SE_VLA_RETREAT_GAIN=0.10 SE_VLA_RETREAT_RELEASE_N=5 ;;
  nocurr) export SE_VLA_SHIELD_OBSERVE_ONLY=1 SE_VLA_POLICY_CHECKPOINT_DIR=$LR/nocurr/ckpt/R2_q00/offset_0 ;;
  nc_shield) : "${SE_VLA_GLM_BASE_URL:=http://HOST_E:8502/v1}"; export SE_VLA_GLM_BASE_URL
         export SE_VLA_POLICY_CHECKPOINT_DIR=$LR/nocurr/ckpt/R2_q00/offset_0
         export SE_VLA_SHIELD_IMPL=vlsa_aegis SE_VLA_VLSA_DOF=3
         unset ZHIPUAI_API_KEY SE_VLA_SHIELD_OBSERVE_ONLY || true ;;
  ncs1) export SE_VLA_SHIELD_OBSERVE_ONLY=1 SE_VLA_POLICY_CHECKPOINT_DIR=$LR/nocurr_s1/ckpt/R2_q00/offset_0 ;;
  ncs2) export SE_VLA_SHIELD_OBSERVE_ONLY=1 SE_VLA_POLICY_CHECKPOINT_DIR=$LR/nocurr_s2/ckpt/R2_q00/offset_0 ;;
  ncs3) export SE_VLA_SHIELD_OBSERVE_ONLY=1 SE_VLA_POLICY_CHECKPOINT_DIR=$LR/nocurr_s3/ckpt/R2_q00/offset_0 ;;
  aegis) : "${SE_VLA_GLM_BASE_URL:=http://HOST_E:8502/v1}"; export SE_VLA_GLM_BASE_URL
         export SE_VLA_SHIELD_IMPL=vlsa_aegis SE_VLA_VLSA_DOF=3
         unset ZHIPUAI_API_KEY SE_VLA_POLICY_CHECKPOINT_DIR SE_VLA_SHIELD_OBSERVE_ONLY || true ;;
esac
if [ "$ARM" != "base" ] && [ "$ARM" != "aegis" ]; then
  [ -d "$SE_VLA_POLICY_CHECKPOINT_DIR/params" ] || { echo "MT_ABORT: no $ARM checkpoint"; exit 1; }
fi
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="multi${TAG}_${ARM}_off${OFF}_r${A}"
echo "=== Multi $TAG(tid=$TID) arm=$ARM off=$OFF adv=$A ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id "$TID" --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((26000 + 3 * (SGE_TASK_ID % 1200))) --array-task-id 1 --server-attempts 3
set -e
[ -s "$OUT/result.json" ] || { echo "MT_FAILED tag=$TAG arm=$ARM off=$OFF adv=$A"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$TAG" "$ARM" "$OFF" "$A" <<'PYV'
import json, re, sys, pathlib
out, tag, arm, off, a = (pathlib.Path(sys.argv[1]),) + tuple(sys.argv[2:6])
r = json.load(open(out/"result.json"))
if r.get("status") != "pass":
    print(f"MT_UNUSABLE tag={tag} arm={arm} off={off} adv={a} status={r.get('status')}"); sys.exit(0)
md = r.get("metric_decomposition") or {}
guarded = None
srd = out/"srd.jsonl"
if srd.exists():
    for line in srd.open(errors="ignore"):
        try: x = json.loads(line)
        except Exception: continue
        if x.get("type") == "vlsa_audit" and x.get("obstacle_name"): guarded = x["obstacle_name"]; break
if arm == "aegis" and guarded is None:
    print(f"MT_UNUSABLE tag={tag} arm={arm} off={off} adv={a} status=shield_did_not_run"); sys.exit(0)
# ETS: the arena logs "Episode finished after N timesteps with cost C"
ets = None
for line in (r.get("log_tail") or []):
    m = re.search(r"after (\d+) timesteps", str(line))
    if m: ets = int(m.group(1)); break
print(f"MT_RESULT tag={tag} arm={arm} off={off} adv={a} "
      f"sr={int((r.get('successes') or 0) > 0)} polcc={float(md.get('policy_induced_cc') or 0)} "
      f"ets={ets} host={(out/'host.txt').read_text().strip()}")
PYV
