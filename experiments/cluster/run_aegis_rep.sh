#!/usr/bin/env bash
#$ -N AEG_rep
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_B
#$ -pe smp 4
#$ -t 1-750
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/aegis_rep/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/aegis_rep/batch/j.$JOB_ID.$TASK_ID.out
#
# AEGIS UNDER THE REPEATED-MEASUREMENT PROTOCOL.
#
# Every AEGIS number this project quotes is a single run (E8, 2026-08-16). Meanwhile a single
# run has been shown to carry almost no information on this benchmark: advancing the sampler
# without changing any executed action flipped 24/24 probe outcomes, and D's own crash count
# ranges 6-16 across five repeats of the identical configuration. So "AEGIS L1t3 crash 3" is
# also one draw, and comparing our 5-repeat mean against it is an unfair asymmetry -- against
# us, since a lucky single draw is hard to beat.
#
# This measures AEGIS the same way we measure ourselves: 50 cells x 5 sampler-advance repeats
# on L1t1 / L1t3 / L1t4. advance=0 must reproduce the recorded E8 numbers, which is the
# identity check.
#
# Shield: SE_VLA_SHIELD_IMPL=vlsa_aegis with dof=3 and self-hosted GLM-4.5V -- the same
# configuration E8 used, verified from run_E8_axfer_local.sh, not the local sphere CBF that
# the SE_VLA_AEGIS_* variable names would suggest.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
V=$LR/aegis_rep
: "${SE_VLA_GLM_BASE_URL:=http://HOST_E:8502/v1}"
export SE_VLA_GLM_BASE_URL
if ! curl -sf --max-time 20 "${SE_VLA_GLM_BASE_URL%/v1}/v1/models" | grep -q glm-4.5v; then
  echo "AEG_ABORT: local GLM not serving glm-4.5v at $SE_VLA_GLM_BASE_URL"; exit 1
fi
unset ZHIPUAI_API_KEY || true
TASKS=(1 3 4); ADV=(0 1 2 3 6)
i=$((SGE_TASK_ID - 1)); ti=$((i / 250)); rem=$((i % 250)); OFF=$((rem / 5)); ai=$((rem % 5))
[ "$ti" -lt 3 ] || { echo "no combo"; exit 0; }
TID=${TASKS[$ti]}; A=${ADV[$ai]}
OUT=$V/L1t${TID}/off${OFF}_r${A}
mkdir -p "$OUT" "$V/batch"
if [ -s "$OUT/result.json" ]; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
unset SE_VLA_POLICY_CHECKPOINT_DIR || true
# vlsa_aegis selects its point-cloud crop box from this; unset it and perceive() dies with
# "no measured crop box for arena suite ''" -- fail-closed upstream, which together with the
# vlsa_ran assertion below stopped a broken run from being recorded as poor AEGIS performance.
export SE_VLA_ARENA_SUITE=safety_static_obstacles
export SE_VLA_SHIELD_IMPL=vlsa_aegis
export SE_VLA_VLSA_DOF=3
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SAMPLER_ADVANCE=$A
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="aegrep_t${TID}_off${OFF}_r${A}"
echo "=== AEGIS L1t$TID off=$OFF advance=$A on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id "$TID" --offset "$OFF" --seed 23 --replan-steps 1 --trials 1 --arm base \
  --port-base $((20000 + 3 * (SGE_TASK_ID % 1500))) --array-task-id 1 --server-attempts 3
set -e
[ -s "$OUT/result.json" ] || { echo "AEG_FAILED t=$TID off=$OFF adv=$A"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$TID" "$OFF" "$A" <<'PYV'
import json, sys, pathlib
out, tid, off, a = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
d = pathlib.Path(out)
r = json.load(open(d/"result.json")); md = r.get("metric_decomposition") or {}
sr = int((r.get("successes") or 0) > 0); cc = float(md.get("policy_induced_cc") or 0)
shield_ok = False
srd = d/"srd.jsonl"
if srd.exists():
    for line in srd.open():
        try: x = json.loads(line)
        except Exception: continue
        if x.get("type") == "vlsa_audit":
            shield_ok = bool(x.get("qp_solved", 0)) and bool(x.get("obstacle_name"))
print(f"AEG_RESULT task=L1t{tid} off={off} adv={a} sr={sr} polcc={cc} vlsa_ran={shield_ok}")
assert shield_ok, "the published shield did not actually run in this cell"
PYV
