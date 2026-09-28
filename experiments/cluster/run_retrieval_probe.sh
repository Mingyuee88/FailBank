#!/usr/bin/env bash
#$ -N retr_probe
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-3
#$ -tc 3
#$ -o ${WORK_ROOT}/lora_dagger/retrieval_probe/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/retrieval_probe/batch/j.$JOB_ID.$TASK_ID.out
# Does the retrieval path work AT ALL once weights are non-zero?
#
# Round 0 produced 502 entries, all weight 0, and corrections_applied stayed 0.
# That is consistent with the credit rule never activating anything, but it does
# NOT prove the credit rule is the only blocker: a downstream query threshold
# could refuse the entries independently. Fixing the credit rule without knowing
# this risks a redesign that still yields zero corrections.
#
# Test: force weight=1.0 on the harvested entries and re-run the SAME offset.
#   task 1: weights forced, shield OFF  -> memory acting alone
#   task 2: weights forced, shield ON   -> memory + shield
#   task 3: weights left at 0 (control) -> must reproduce corrections_applied=0
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers PI05_SERVER_TIMEOUT_SEC=600
cd ${SE_VLA_ROOT}

ROOT=${WORK_ROOT}/lora_dagger/retrieval_probe
SRC=${WORK_ROOT}/lora_dagger/round0_control/aegis-adaptive/final_checkpoint.json
mkdir -p "$ROOT/batch"

case $SGE_TASK_ID in
  1) NAME=forced_shieldoff; W=1.0; AEGIS=0 ;;
  2) NAME=forced_shieldon;  W=1.0; AEGIS=1 ;;
  3) NAME=zero_control;     W=0.0; AEGIS=0 ;;
esac

CK=$ROOT/$NAME/checkpoint.json
OUT=$ROOT/$NAME
mkdir -p "$OUT"

python3 - "$SRC" "$CK" "$W" <<'PY'
import json, sys
src, dst, w = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = json.load(open(src))
ents = d["relation_memory"]["entries"]
for e in ents:
    e["weight"] = w
    e["srd_status"] = "forced_retrieval_probe"
d["metadata"] = {"saved_by": "retrieval_probe",
                 "note": f"weights forced to {w} to test the retrieval path only"}
import os
os.makedirs(os.path.dirname(dst), exist_ok=True)
open(dst, "w").write(json.dumps(d, indent=2, sort_keys=True) + "\n")
print(f"wrote {dst}: {len(ents)} entries at weight {w}")
PY

export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_EPISODE_ID="retr_${NAME}"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_SRD_WRITES_ENABLED=0 SE_VLA_SRD_ENABLED=1
export SE_VLA_RELATION_MEMORY_FEATURE_KEYS="cost_pair_min_distance,cost_pair_delta_xy[0],cost_pair_delta_xy[1],fall_max_angle_abs,episode_step_index"
export SE_VLA_POLICY_CHECKPOINT_DIR=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
export SE_VLA_AEGIS_SOURCE=$AEGIS
export SE_VLA_AEGIS_ALPHA=3 SE_VLA_AEGIS_EEF_RADIUS=0.03 SE_VLA_AEGIS_ORACLE_RADIUS=0.04 SE_VLA_AEGIS_MARGIN=0.02

echo "=== retrieval probe $NAME weight=$W aegis=$AEGIS ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 2 --task-id 4 --offset 10 --seed 23 \
  --replan-steps 1 --trials 1 --arm pruned_memory --checkpoint "$CK" \
  --port-base $((40000 + SGE_TASK_ID)) --array-task-id "$SGE_TASK_ID" --server-attempts 3
echo "RETR_EXIT=$? name=$NAME"
