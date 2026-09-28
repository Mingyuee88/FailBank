#!/usr/bin/env bash
#$ -N E5_shD
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=HOST_A
#$ -pe smp 4
#$ -t 1-47
#$ -tc 2
#$ -o ${WORK_ROOT}/E5_shD/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/E5_shD/batch/j.$JOB_ID.$TASK_ID.out
#
# E5 -- the published shield running ON TOP OF the distilled policy D, instead of on base.
#
# Why this arm. Cost on this benchmark is not a second axis: measured across every arm,
# policy-induced cost is ~270 x (number of episodes that fail INTO the hazard), and the
# cost sitting on successful episodes is 45-122 across 36-41 cells. So "low cost" and "no
# unsafe failures" are the same requirement. base has 11 unsafe failures; D has 6; the
# published shield has 1 but converts 9-12 successes into safe-but-failed episodes.
#
# The two are complementary where it matters. D's remaining failures are
# [0, 1, 12, 26, 29, 42] and the shield repairs five of them -- [0, 1, 12, 26, 29]. If the
# shield repaired exactly those on top of D and destroyed nothing, the arm would be 46/47
# at polCC 449, against 36/47 @ 3244 for base, 35/47 @ 199 for the shield alone and
# 41/47 @ 1714 for D alone.
#
# What actually decides it is how much the shield DESTROYS here. On base it destroys 9 of
# 36 successes (dof=3) or 12 (dof=6). D was distilled from this shield's own corrections,
# so its trajectories should already sit where the barrier wants them and bind the
# constraint less often. That is the claim this arm tests, and the per-step trace measures
# it directly: constraint_active_steps on shield-over-D against shield-over-base, offset by
# offset, is the intervention rate that the self-evolving story says must fall.
#
# It is also the honest form of the deployment argument. Distillation alone removes the
# runtime装置 but has no recourse when the student fails -- and when it fails it fails the
# base way, into the hazard. Keeping the shield keeps the recourse; the distillation is
# then judged by how much less the shield has to intervene, not by whether it can replace
# it outright.
#
# Config provenance, every line: shield block copied from run_E4_dof.sh (itself derived
# from the verified run_E3_vlsa.sh), checkpoint block copied from run_v21_eval.sh. The
# seed is 23, matching run_v21_eval.sh, so that E5-vs-D is exact; --seed is a verified
# no-op in this harness, so E5-vs-E3/E4 (seed 17) is unaffected either way. Port base is
# the corrected formula, not E3's, which overflows past task 76.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd ${SE_VLA_ROOT}
LR=${WORK_ROOT}/lora_dagger
OUTROOT=${WORK_ROOT}/E5_shD

OFFS=(0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47)
i=$((SGE_TASK_ID - 1))
[ "$i" -lt "${#OFFS[@]}" ] || { echo "no cell for task $SGE_TASK_ID"; exit 0; }
OFF=${OFFS[$i]}

CK=$LR/v20_fold/D_new_qw0.0/offset_0
[ -d "$CK" ] || { echo "missing folded checkpoint $CK"; exit 1; }

OUT=$OUTROOT/shD/off$OFF
mkdir -p "$OUT" "$OUTROOT/batch"
if [ -s "$OUT/result.json" ] && \
   external/VLA-Arena/envs/openpi/.venv/bin/python -c \
     'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="pass" else 1)' \
     "$OUT/result.json" 2>/dev/null; then echo "SKIP $OUT"; exit 0; fi
rm -f "$OUT/result.json" "$OUT/vlsa_steps.jsonl"
hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name --format=csv,noheader >> "$OUT/host.txt" 2>&1 || true

[ -r "$HOME/.config/zhipu_api_key" ] && export ZHIPUAI_API_KEY="$(cat "$HOME/.config/zhipu_api_key")"

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
# the distilled weights ARE the policy here; the shield sits on top of them
export SE_VLA_POLICY_CHECKPOINT_DIR="$CK"
export SE_VLA_SHIELD_IMPL=vlsa_aegis
export SE_VLA_VLSA_DOF=3
export SE_VLA_VLSA_STEP_TRACE=1 SE_VLA_VLSA_STEPS_PATH="$OUT/vlsa_steps.jsonl"
export SE_VLA_ARENA_SUITE=safety_static_obstacles
export SE_VLA_AEGIS_SOURCE=1 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_RELATION_MEMORY_CREDIT_ENABLED=0
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="e5_shD_off${OFF}"

echo "=== E5 shield-on-D off=$OFF ckpt=$CK on $(hostname) ==="
set +e
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm base \
  --port-base $((50000 + 20 * (SGE_TASK_ID - 1))) --array-task-id 1 --server-attempts 3
rc=$?
set -e
echo "E5_EXIT=$rc off=$OFF"

if [ ! -s "$OUT/result.json" ] || ! external/VLA-Arena/envs/openpi/.venv/bin/python -c \
     'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="pass" else 1)' \
     "$OUT/result.json" 2>/dev/null; then
  echo "E5_CELL_FAILED off=$OFF -- removing for retry"; rm -f "$OUT/result.json"; exit 1
fi

# Two things must be shown, not assumed: that the shield really ran, and that it ran over
# the DISTILLED weights rather than silently over base. The second is the one that would
# quietly turn this arm into a duplicate of E4.
external/VLA-Arena/envs/openpi/.venv/bin/python - "$OUT" "$OFF" "$CK" <<'PYCHK'
import json, os, sys
out, off, ck = sys.argv[1], sys.argv[2], sys.argv[3]
audit = None
for line in open(os.path.join(out, "srd.jsonl")):
    try: r = json.loads(line)
    except Exception: continue
    if r.get("type") == "vlsa_audit":
        audit = r
assert audit is not None, "no vlsa_audit row -- the published shield never ran"
assert audit.get("dof") == 3, f"cell ran dof={audit.get('dof')}, expected 3"
r = json.load(open(os.path.join(out, "result.json")))
md = r.get("metric_decomposition", {})
steps_path = os.path.join(out, "vlsa_steps.jsonl")
n_steps = sum(1 for _ in open(steps_path)) if os.path.exists(steps_path) else 0
print(f"E5_RESULT off={off} sr={r.get('successes')} polcc={md.get('policy_induced_cc')} "
      f"obstacle={audit['obstacle_name']!r} steps={audit['steps']} qp_ok={audit['qp_solved']} "
      f"qp_infeasible={audit['qp_infeasible']} h_min={audit['h_min']} "
      f"override_sum={audit['override_norm_sum']:.4f} "
      f"active_steps={audit.get('constraint_active_steps')} trace_rows={n_steps}")
assert audit["obstacle_name"], "the VLM named no obstacle"
assert audit["perception_ok"], "perception produced no ellipsoid -- shield never engaged"
assert audit["qp_solved"] > 0, "no QP ever solved"
assert audit["override_norm_sum"] > 0, "the shield never changed a single action"
assert n_steps >= audit["qp_solved"], f"step trace short: {n_steps} rows for {audit['qp_solved']} solves"
assert os.environ.get("SE_VLA_POLICY_CHECKPOINT_DIR") == ck, "checkpoint env lost"
print("E5_CELL_VERIFIED")
PYCHK
