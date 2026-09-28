#!/usr/bin/env bash
#$ -N RvEval
#$ -cwd
#$ -l gpu_card=1
#$ -pe smp 4
#$ -j y
# Review 2026-09-17 E14-C copy of eval.sh: only change is HZ_EEF_RADIUS / HZ_MARGIN overrides (defaults identical).
# Review 2026-09-13. One arm on one task. The environment block is the union of
# run_multitask.sh / run_l2_task.sh / run_tau_eval_t3.sh / run_dyn_r3.sh, copied verbatim;
# only the arm wiring is parameterised.
# Required -v: ARM CK SUITE LEVEL TID NOFF REPS OUTROOT EXPECT_HOST PORTBASE SHIELD EXPORT_SUITE
#   CK "-" means base weights.
#   SHIELD: observe (SHIELD_OBSERVE_ONLY=1), inloop_local (the labelling teacher executed),
#           aegis (vlsa_aegis with GLM-4.5V perception).
#   EXPORT_SUITE 1|0 mirrors the source launcher (run_tau_eval_t3/t4 did not export
#           SE_VLA_ARENA_SUITE; run_multitask / run_l2_task / run_dyn_r3 did).
#   SE_VLA_SAMPLER_ADVANCE is exported only when REPS>1 (value 0 is the code default).
# Optional -v: PI0=1 (pi0 yaml + config resolution), TIMING=1 (wall clock into timing.txt).
set -euo pipefail
for v in ARM CK SUITE LEVEL TID NOFF REPS OUTROOT EXPECT_HOST PORTBASE SHIELD EXPORT_SUITE; do
  eval "test -n \"\${$v:-}\"" || { echo "MISSING_PARAM $v"; exit 1; }
done
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
i=$((SGE_TASK_ID - 1)); OFF=$((i / REPS)); A=$((i % REPS))
if [ "$OFF" -ge "$NOFF" ]; then echo "no cell"; exit 0; fi
OUT=$OUTROOT/$ARM/off${OFF}_r${A}
mkdir -p "$OUT"
if [ -s "$OUT/result.json" ] && grep -q '"status": "pass"' "$OUT/result.json"; then echo "SKIP $OUT"; exit 0; fi
hostname > "$OUT/host.txt"
if [ "$(hostname -s)" != "$EXPECT_HOST" ]; then
  echo "HOST_MISMATCH expected=$EXPECT_HOST got=$(hostname -s) cell=$OUT"
  rm -f "$OUT/result.json"; exit 1
fi
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
if [ "$EXPORT_SUITE" = "1" ]; then export SE_VLA_ARENA_SUITE="$SUITE"; fi
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=${HZ_EEF_RADIUS:-0.03} SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=${HZ_MARGIN:-0.02}
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
if [ "$REPS" -gt 1 ]; then export SE_VLA_SAMPLER_ADVANCE=$A; fi
if [ "$SUITE" = "safety_dynamic_obstacles" ]; then export SE_VLA_HAZARD_LOOKAHEAD_S=0.0; fi
unset SE_VLA_POLICY_CHECKPOINT_DIR SE_VLA_SHIELD_OBSERVE_ONLY SE_VLA_SHIELD_IMPL || true
case "$SHIELD" in
  observe)      export SE_VLA_SHIELD_OBSERVE_ONLY=1 ;;
  inloop_local) export SE_VLA_SHIELD_IMPL=local ;;
  aegis)        : "${SE_VLA_GLM_BASE_URL:=http://HOST_E:8502/v1}"; export SE_VLA_GLM_BASE_URL
                export SE_VLA_SHIELD_IMPL=vlsa_aegis SE_VLA_VLSA_DOF=3
                unset ZHIPUAI_API_KEY || true ;;
  *) echo "BAD_SHIELD $SHIELD"; exit 1 ;;
esac
if [ "$CK" != "-" ]; then
  [ -d "$CK/params" ] || { echo "EVAL_ABORT no params at $CK"; exit 1; }
  export SE_VLA_POLICY_CHECKPOINT_DIR="$CK"
fi
if [ -n "${PI0:-}" ]; then
  export SE_VLA_OPENPI_YAML=${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/configs/evaluation/openpi_pi0.yaml
  export SE_VLA_CONFIG_RESOLUTION=${SE_VLA_ROOT}/roundG/pi05_smoke/config_resolution_pi0.json
fi
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="rv0913_${ARM}_L${LEVEL}t${TID}_off${OFF}_r${A}"
echo "=== RvEval arm=$ARM ck=$CK suite=$SUITE L$LEVEL t$TID off=$OFF adv=$A shield=$SHIELD on $(hostname) ==="
t0=$(date +%s.%N)
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite "$SUITE" \
  --task-level "$LEVEL" --task-id "$TID" --offset "$OFF" --seed 17 --replan-steps 1 --trials 1 --arm base \
  --port-base $((PORTBASE + 20 * i)) --array-task-id 1 --server-attempts 3
rc=$?
set -e
t1=$(date +%s.%N)
if [ "${TIMING:-0}" = "1" ]; then echo "$t0 $t1" > "$OUT/timing.txt"; fi
echo "EVAL_EXIT=$rc"
[ -s "$OUT/result.json" ] || { echo "EVAL_FAILED $OUT"; exit 1; }
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$CK" "$ARM" <<"PY"
import json, sys, glob, os, re
out, ck, arm = sys.argv[1:4]
r = json.load(open(os.path.join(out, "result.json")))
if r.get("status") != "pass":
    print("EVAL_UNUSABLE arm=%s status=%s" % (arm, r.get("status")))
    sys.exit(0)
restored = None
for lg in glob.glob(os.path.join(out, "server_port*.log")):
    for line in open(lg, errors="ignore"):
        if "Finished restoring checkpoint" in line:
            restored = line.strip().split(" from ")[-1].rstrip(".")
            break
    if restored:
        break
if ck != "-":
    want = ck.rstrip("/") + "/params"
    if restored != want:
        print("PROVENANCE_MISMATCH arm=%s restored=%s want=%s" % (arm, restored, want))
        sys.exit(1)
elif restored is None or "/checkpoints/" not in restored:
    print("PROVENANCE_MISMATCH arm=%s restored=%s want=<base checkpoint>" % (arm, restored))
    sys.exit(1)
fam = (r.get("architecture") or {}).get("architecture_family")
if os.environ.get("PI0") and fam != "pi0":
    os.rename(os.path.join(out, "result.json"), os.path.join(out, "result.family_mismatch.json"))
    print("FAMILY_MISMATCH arm=%s family=%s" % (arm, fam))
    sys.exit(1)
md = r.get("metric_decomposition") or {}
ad = r.get("adapter") or {}
ets = None
for line in (r.get("log_tail") or []):
    m = re.search(r"after (\d+) timesteps", str(line))
    if m:
        ets = int(m.group(1)); break
print("EVAL_RESULT arm=%s sr=%s offcc=%s polcc=%s corrections=%s fwd_unchanged=%s ets=%s restored=%s" % (
    arm, r.get("successes"), md.get("official_cc"), md.get("policy_induced_cc"),
    ad.get("corrections_applied"), ad.get("all_actions_forwarded_unchanged"), ets, restored))
PY
