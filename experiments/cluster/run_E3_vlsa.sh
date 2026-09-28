#!/usr/bin/env bash
#$ -N E3_vlsa
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 4
#$ -t 1-47
#$ -tc 2
#$ -o ${WORK_ROOT}/E3_vlsa/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/E3_vlsa/batch/j.$JOB_ID.$TASK_ID.out
#
# E3 -- the PUBLISHED AEGIS shield, its own code, running inside the VLA-Arena harness.
#
# E1 established that we can run their method: on SafeLIBERO-Spatial our run of their
# repository gave CAR 77.8 [73.4,81.6] against their published 75.5, TSR 72.5 [67.9,76.6]
# against 73.3, ETS 187.7 against 188.2, and the shield EFFECT over a locally-run base
# matched theirs almost exactly (dCAR +65.5 vs +60.2, dTSR +8.8 vs +13.5). So the pipeline
# is faithful and the direction is theirs: AEGIS raises task success.
#
# This runs that same shield here. Everything except the shield is the machinery every
# other arm in this study used -- same policy server, same 47 offsets, same cost
# accounting, same result.json -- so base / published AEGIS / our method differ only in
# which function computes the executed action.
#
# Ported seams, each measured rather than assumed:
#   backview camera   copied from the SafeLIBERO scene of the same workspace family, only
#                     after agentview/birdview/sideview were confirmed identical between
#                     the two repositories (sideview to 16 decimals), so the world frames
#                     coincide. Verified at run time: depth 100% finite, 6468 distinct
#                     values, the two views differ by 49.9 mean pixel intensity.
#   crop box          their filtering_points selects by suite-name substring and knows only
#                     LIBERO's names. Arena tabletop was measured against theirs: table top
#                     z=0.90 in both, their box floor 0.92 = table + 2 cm, arena hazard
#                     geoms span 0.909-1.134, so the table box transfers. hazard_avoidance
#                     is NOT mapped -- its flat stove sits at z=0.905 and the same box
#                     would erase it, yielding a shield that never engages and a "no
#                     effect" result that would be an artefact of the crop.
#   shield functions  imported unchanged from vlsa-aegis/main/utils.py; only the glue from
#                     main_aegis_translational.py was rebuilt, with every constant cited.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
OUTROOT=${WORK_ROOT}/E3_vlsa

OFFS=(0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47)
i=$((SGE_TASK_ID - 1))
[ "$i" -lt "${#OFFS[@]}" ] || { echo "no cell for task $SGE_TASK_ID"; exit 0; }
OFF=${OFFS[$i]}

OUT=$OUTROOT/aegis/off$OFF
mkdir -p "$OUT" "$OUTROOT/batch"
if [ -s "$OUT/result.json" ] && \
   external/VLA-Arena/envs/openpi/.venv/bin/python -c \
     'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="pass" else 1)' \
     "$OUT/result.json" 2>/dev/null; then echo "SKIP $OUT"; exit 0; fi
rm -f "$OUT/result.json"
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

# The VLM key is read at run time from a 600-mode file; it never enters a script or argv.
[ -r "$HOME/.config/zhipu_api_key" ] && export ZHIPUAI_API_KEY="$(cat "$HOME/.config/zhipu_api_key")"

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
unset SE_VLA_POLICY_CHECKPOINT_DIR || true
export SE_VLA_SHIELD_IMPL=vlsa_aegis
export SE_VLA_ARENA_SUITE=safety_static_obstacles
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SEED=17 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="e3_vlsa_off${OFF}"

echo "=== E3 published-AEGIS off=$OFF on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 17 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((64000 + 20 * SGE_TASK_ID)) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "E3_EXIT=$rc off=$OFF"

if [ ! -s "$OUT/result.json" ] || ! external/VLA-Arena/envs/openpi/.venv/bin/python -c \
     'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="pass" else 1)' \
     "$OUT/result.json" 2>/dev/null; then
  echo "E3_CELL_FAILED off=$OFF -- removing for retry"; rm -f "$OUT/result.json"; exit 1
fi

# The port must be shown to have PERCEIVED and SOLVED, not merely to have run. A silent
# degradation -- no obstacle named, empty cloud, every QP infeasible -- would otherwise
# look like a clean null result for their method.
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$OFF" <<'PYCHK'
import json, os, sys
out, off = sys.argv[1], sys.argv[2]
audit = None
for line in open(os.path.join(out, "srd.jsonl")):
    try: r = json.loads(line)
    except Exception: continue
    if r.get("type") == "vlsa_audit":
        audit = r
assert audit is not None, "no vlsa_audit row -- the published shield never ran"
r = json.load(open(os.path.join(out, "result.json")))
md = r.get("metric_decomposition", {})
print(f"E3_RESULT off={off} sr={r.get('successes')} polcc={md.get('policy_induced_cc')} "
      f"obstacle={audit['obstacle_name']!r} pts a/b/fused/filt="
      f"{audit['points_agentview']}/{audit['points_backview']}/{audit['points_fused']}/{audit['points_filtered']} "
      f"steps={audit['steps']} qp_ok={audit['qp_solved']} qp_infeasible={audit['qp_infeasible']} "
      f"h_min={audit['h_min']} override_sum={audit['override_norm_sum']:.4f}")
assert audit["obstacle_name"], "the VLM named no obstacle"
assert audit["perception_ok"], "perception produced no ellipsoid -- shield never engaged"
assert audit["qp_solved"] > 0, "no QP ever solved"
assert audit["override_norm_sum"] > 0, "the shield never changed a single action"
print("E3_CELL_VERIFIED")
PYCHK
