#!/usr/bin/env bash
#$ -N p3b_resc
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-40
#$ -tc 6
#$ -o ${WORK_ROOT}/lora_dagger/phase3b_retention/batch/resc.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/phase3b_retention/batch/resc.$JOB_ID.$TASK_ID.out
#
# Re-run cells that died with "policy server did not become ready in 600s". That is
# NFS contention at model load (four tasks pulling a 12 GB checkpoint at once off a
# filesystem at 94%), not a scientific outcome -- fail_cell_atomically() discarded
# the partial products, so the cell is simply missing and must be redone.
#   * server timeout raised 600 -> 1800 s
#   * concurrency 6, and this job is held until the producers finish, so the set of
#     failed cells is stable and every task derives the same deterministic list.
# Only Phase3b lora/base cells are rescued here; repeat-eval cells are listed too.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers
export PI05_SERVER_TIMEOUT_SEC=1800
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
cd ${SE_VLA_ROOT}

LR=${WORK_ROOT}/lora_dagger
LINE=$(python3 - "$SGE_TASK_ID" <<'PY'
import json, glob, os, re, sys
idx = int(sys.argv[1])
LR = "${WORK_ROOT}/lora_dagger"
bad = []
for d in sorted(glob.glob(LR + "/phase3b_retention/lora/off*_f*")) + \
         sorted(glob.glob(LR + "/phase3b_retention/base/off*")):
    p = os.path.join(d, "result.json")
    if not os.path.exists(p):
        continue
    try:
        r = json.load(open(p))
    except Exception:
        r = {}
    if "successes" in r:
        continue
    m = re.search(r"off(\d+)_f(\d+)$", d)
    if m:
        bad.append("lora %s %s" % (m.group(1), m.group(2)))
    else:
        bad.append("base %s -1" % re.search(r"off(\d+)$", d).group(1))
print(bad[idx - 1] if idx <= len(bad) else "")
PY
)
[ -n "$LINE" ] || { echo "no failed cell at index $SGE_TASK_ID (nothing to rescue)"; exit 0; }
ARM=$(echo "$LINE" | awk '{print $1}')
OFF=$(echo "$LINE" | awk '{print $2}')
FOLD=$(echo "$LINE" | awk '{print $3}')

if [ "$ARM" = "lora" ]; then
  CKPT=$LR/phase2_training/merged/offset_${FOLD}
  OUT=$LR/phase3b_retention/lora/off${OFF}_f${FOLD}
else
  CKPT=${SE_VLA_ROOT}/checkpoints/pi05_vla_arena_finetuned
  OUT=$LR/phase3b_retention/base/off${OFF}
fi
mkdir -p "$OUT"
export PI05_SERVER_LOG_DIR="$OUT" SE_VLA_SRD_TELEMETRY_PATH="$OUT/srd.jsonl"
export SE_VLA_SEED=23 SE_VLA_RESULT_PATH="$OUT/result.json"
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
export SE_VLA_SRD_WRITES_ENABLED=0
unset SE_VLA_PHASE1_RECORD || true
export SE_VLA_POLICY_CHECKPOINT_DIR="$CKPT"
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=0

echo "=== RESCUE arm=$ARM offset=$OFF fold=$FOLD ==="
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$OUT/result.json" --task-suite safety_static_obstacles \
  --task-level 1 --task-id 2 --offset "$OFF" --seed 23 \
  --replan-steps 1 --trials 1 --arm base --port-base 52000 \
  --array-task-id "$SGE_TASK_ID" --server-attempts 3
echo "RESCUE_EXIT=$? arm=$ARM off=$OFF fold=$FOLD"
