#!/usr/bin/env bash
#$ -N E7_concord
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 4
#$ -t 1-47
#$ -tc 2
#$ -o ${WORK_ROOT}/E7_concord/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/E7_concord/batch/j.$JOB_ID.$TASK_ID.out
#
# E7 -- the hard gate on self-hosted perception.
#
# Re-runs E4/dof3 exactly, changing ONE thing: obstacle_detection() talks to a locally
# served GLM-4.5V instead of the hosted Zhipu endpoint. Same shield, same variant, same
# offsets, same policy, same host pin, same seed. Per-cell success and policy-induced cost
# must reproduce E4/dof3, which itself already reproduced E3 bit-for-bit on the 29 cells
# they share.
#
# Why this is a gate and not a formality. If it passes, the perception stage stops
# depending on a metered, versioned-by-the-vendor endpoint and every downstream arm --
# the E6 backfill, the teacher swap, the >=6-seed attribution ladder -- becomes affordable
# and replayable. If it FAILS, that is also a result worth having: it would mean the
# published method's outcomes depend on which GLM the vendor happens to be serving, which
# belongs in the reproducibility section rather than being papered over. Either way the
# answer gets reported; it is not re-run until it agrees.
#
# What must NOT happen is adopting a cheaper model because it is convenient. If GLM-4.5V
# cannot be served, the fallbacks (bitsandbytes 4-bit, or GLM-4.1V-9B) change the
# perception stage and may only be adopted on evidence from THIS check, never to save time.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
OUTROOT=${WORK_ROOT}/E7_concord

# The local VLM. Fail closed: a cell that silently fell back to the hosted endpoint (or to
# no endpoint at all) would produce a comparison that means nothing.
: "${SE_VLA_GLM_BASE_URL:=http://HOST_E:8502/v1}"
export SE_VLA_GLM_BASE_URL
if ! curl -sf --max-time 20 "${SE_VLA_GLM_BASE_URL%/v1}/v1/models" | grep -q 'glm-4.5v'; then
  echo "E7_ABORT: local GLM not serving glm-4.5v at $SE_VLA_GLM_BASE_URL"; exit 1
fi
unset ZHIPUAI_API_KEY || true   # nothing may reach the hosted endpoint from this arm

OFFS=(0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47)
i=$((SGE_TASK_ID - 1))
[ "$i" -lt "${#OFFS[@]}" ] || { echo "no cell for task $SGE_TASK_ID"; exit 0; }
OFF=${OFFS[$i]}

OUT=$OUTROOT/local/off$OFF
mkdir -p "$OUT" "$OUTROOT/batch"
if [ -s "$OUT/result.json" ] && \
   external/VLA-Arena/envs/openpi/.venv/bin/python -c \
     'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="pass" else 1)' \
     "$OUT/result.json" 2>/dev/null; then echo "SKIP $OUT"; exit 0; fi
rm -f "$OUT/result.json" "$OUT/vlsa_steps.jsonl"
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
unset SE_VLA_POLICY_CHECKPOINT_DIR || true
export SE_VLA_SHIELD_IMPL=vlsa_aegis
export SE_VLA_VLSA_DOF=3
export SE_VLA_VLSA_STEP_TRACE=1 SE_VLA_VLSA_STEPS_PATH="$OUT/vlsa_steps.jsonl"
export SE_VLA_ARENA_SUITE=safety_static_obstacles
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="e7_local_off${OFF}"

echo "=== E7 local-GLM dof=3 off=$OFF on $(hostname) via $SE_VLA_GLM_BASE_URL ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((50000 + 20 * (SGE_TASK_ID - 1))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "E7_EXIT=$rc off=$OFF"

if [ ! -s "$OUT/result.json" ] || ! external/VLA-Arena/envs/openpi/.venv/bin/python -c \
     'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="pass" else 1)' \
     "$OUT/result.json" 2>/dev/null; then
  echo "E7_CELL_FAILED off=$OFF -- removing for retry"; rm -f "$OUT/result.json"; exit 1
fi

external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$OFF" <<'PYCHK'
import json, os, sys
out, off = sys.argv[1], sys.argv[2]
audit = None
for line in open(os.path.join(out, "srd.jsonl")):
    try: r = json.loads(line)
    except Exception: continue
    if r.get("type") == "vlsa_audit":
        audit = r
assert audit is not None, "no vlsa_audit row -- the shield never ran"
assert audit.get("dof") == 3, f"cell ran dof={audit.get('dof')}, expected 3"
r = json.load(open(os.path.join(out, "result.json")))
md = r.get("metric_decomposition", {})
print(f"E7_RESULT off={off} sr={r.get('successes')} polcc={md.get('policy_induced_cc')} "
      f"obstacle={audit['obstacle_name']!r} pts_filt={audit['points_filtered']} "
      f"steps={audit['steps']} qp_ok={audit['qp_solved']} h_min={audit['h_min']} "
      f"override_sum={audit['override_norm_sum']:.4f} "
      f"retries={audit.get('perception_retries')}")
assert audit["obstacle_name"], "the VLM named no obstacle"
# The failure this really guards: a served build without --reasoning-parser returns the
# whole chain of thought, GroundingDINO grounds on a paragraph, and every point cloud is
# quietly wrong while nothing crashes.
name = str(audit["obstacle_name"])
assert len(name) <= 40 and "\n" not in name, f"obstacle name looks like a reasoning chain: {name[:200]!r}"
assert audit["perception_ok"], "perception produced no ellipsoid"
assert audit["qp_solved"] > 0, "no QP ever solved"
assert audit["override_norm_sum"] > 0, "the shield never changed an action"
print("E7_CELL_VERIFIED")
PYCHK
