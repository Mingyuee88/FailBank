#!/usr/bin/env bash
#$ -N MotLemC
#$ -cwd
#$ -q gpu@HOST_A
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-360
#$ -tc 4
#$ -j y

# Confirmation-only offsets 10--49: 3 arms x 40 offsets x 3 sampler advances.
# CANDIDATE must be the single variant chosen by check_lemon_screen.py.
set -euo pipefail
: "${CANDIDATE:?set CANDIDATE=q20 or q20_r3 from the development gate}"
case "$CANDIDATE" in
  q20) CKNAME=R2_q20 ;;
  q20_r3) CKNAME=R3c_q20 ;;
  *) echo "ABORT: invalid candidate $CANDIDATE"; exit 1 ;;
esac
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
ROOT=${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
CK=$LR/motivation_search/checkpoints/$CKNAME/offset_0
OUTROOT=$LR/motivation_search/lemon_l2_confirm_$CANDIDATE
EXPECT_HOST=HOST_A
ARMS=(base aegis failbank); ADV=(0 1 2)
i=$((SGE_TASK_ID - 1)); ai=$((i / 120)); rem=$((i % 120)); OFF=$((10 + rem / 3)); A=${ADV[$((rem % 3))]}; ARM=${ARMS[$ai]}
OUT=$OUTROOT/$ARM/off${OFF}_r${A}
cd "$ROOT"
mkdir -p "$OUT" "$OUTROOT/batch"
[ -d "$CK/params" ] || { echo "ABORT: missing selected checkpoint $CK"; exit 1; }
if [ -s "$OUT/result.json" ] && grep -q '"status": "pass"' "$OUT/result.json"; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
[ "$(hostname -s)" = "$EXPECT_HOST" ] || { rm -f "$OUT/result.json"; echo HOST_MISMATCH; exit 1; }
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_ARENA_SUITE=safety_static_obstacles
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0 SE_VLA_HAZARD_LOOKAHEAD_S=0.0
export SE_VLA_SAMPLER_ADVANCE="$A"
unset SE_VLA_SHIELD_IMPL SE_VLA_VLSA_DOF SE_VLA_GLM_BASE_URL SE_VLA_POLICY_CHECKPOINT_DIR SE_VLA_SHIELD_OBSERVE_ONLY || true
case "$ARM" in
  base) export SE_VLA_SHIELD_OBSERVE_ONLY=1 ;;
  failbank) export SE_VLA_SHIELD_OBSERVE_ONLY=1 SE_VLA_POLICY_CHECKPOINT_DIR="$CK" ;;
  aegis) export SE_VLA_GLM_BASE_URL=http://HOST_E:8502/v1
         export SE_VLA_SHIELD_IMPL=vlsa_aegis SE_VLA_VLSA_DOF=3
         unset ZHIPUAI_API_KEY || true ;;
esac
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="motiv_lemon_confirm_${CANDIDATE}_${ARM}_off${OFF}_r${A}"
echo "LEMON_CONFIRM_START candidate=$CANDIDATE arm=$ARM off=$OFF advance=$A host=$(hostname -s)"
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 2 --task-id 1 --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((33000 + 3 * (SGE_TASK_ID % 1000))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
[ "$rc" -eq 0 ] && [ -s "$OUT/result.json" ] || { rm -f "$OUT/result.json"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$ARM" "$CK" <<'PYV'
import json, pathlib, sys
out, arm, ck = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
r = json.load(open(out / "result.json")); ti = r.get("task_identity") or {}; ad = r.get("adapter") or {}
ok = r.get("status") == "pass" and ti.get("identity_gate") == "pass"
ok &= ti.get("configured_level") == 2 and ti.get("configured_task_id") == 1
if arm in {"base", "failbank"}: ok &= ad.get("all_actions_forwarded_unchanged") is True
if arm == "failbank":
    ok &= any(ck in p.read_text(errors="ignore") for p in out.glob("server_port*.log"))
if arm == "aegis":
    p = out / "srd.jsonl"
    ok &= p.exists() and any('"type": "vlsa_audit"' in line for line in p.open())
if not ok:
    (out / "result.json").unlink(missing_ok=True)
    raise SystemExit(f"UNUSABLE arm={arm}")
md = r.get("metric_decomposition") or {}
print(f"LEMON_CONFIRM_PASS arm={arm} success={r.get('successes')} official_cc={md.get('official_cc')}")
PYV
