#!/usr/bin/env bash
#$ -N MotLemR3
#$ -cwd
#$ -q gpu@HOST_A
#$ -l gpu_card=1
#$ -pe smp 4
#$ -t 1-30
#$ -tc 4
#$ -j y

# Development-only R3 candidate on the same 10 offsets x 3 advances as R2.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
ROOT=${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
CK=$LR/motivation_search/checkpoints/R3c_q20/offset_0
OUTROOT=$LR/motivation_search/lemon_l2_screen/q20_r3
EXPECT_HOST=HOST_A
ADV=(0 1 2)
i=$((SGE_TASK_ID - 1)); OFF=$((i / 3)); A=${ADV[$((i % 3))]}
OUT=$OUTROOT/off${OFF}_r${A}
cd "$ROOT"
mkdir -p "$OUT" "$OUTROOT/batch"
[ -d "$CK/params" ] || { echo "ABORT: missing R3 checkpoint $CK"; exit 1; }
if [ -s "$OUT/result.json" ] && grep -q '"status": "pass"' "$OUT/result.json"; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
[ "$(hostname -s)" = "$EXPECT_HOST" ] || { rm -f "$OUT/result.json"; echo HOST_MISMATCH; exit 1; }
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_ARENA_SUITE=safety_static_obstacles
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0 SE_VLA_HAZARD_LOOKAHEAD_S=0.0
export SE_VLA_SHIELD_OBSERVE_ONLY=1 SE_VLA_POLICY_CHECKPOINT_DIR="$CK"
export SE_VLA_SAMPLER_ADVANCE="$A" SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="motiv_lemon_screen_q20_r3_off${OFF}_r${A}"
echo "LEMON_R3_SCREEN_START host=$(hostname -s) off=$OFF advance=$A checkpoint=$CK"
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 2 --task-id 1 --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((29000 + 3 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
rc=$?
set -e
[ "$rc" -eq 0 ] && [ -s "$OUT/result.json" ] || { rm -f "$OUT/result.json"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$CK" <<'PYV'
import json, pathlib, sys
out, ck = pathlib.Path(sys.argv[1]), sys.argv[2]
r = json.load(open(out / "result.json")); ti = r.get("task_identity") or {}; ad = r.get("adapter") or {}
logs = list(out.glob("server_port*.log"))
ok = r.get("status") == "pass" and ti.get("identity_gate") == "pass"
ok &= ti.get("configured_level") == 2 and ti.get("configured_task_id") == 1
ok &= ad.get("all_actions_forwarded_unchanged") is True
ok &= any(ck in p.read_text(errors="ignore") for p in logs)
if not ok:
    (out / "result.json").unlink(missing_ok=True)
    raise SystemExit("UNUSABLE q20_r3")
md = r.get("metric_decomposition") or {}
print(f"LEMON_R3_SCREEN_PASS success={r.get('successes')} official_cc={md.get('official_cc')}")
PYV
