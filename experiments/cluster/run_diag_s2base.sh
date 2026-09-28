#!/usr/bin/env bash
#$ -N diag_s2base
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-4
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/diag_stage2base/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/diag_stage2base/batch/j.$JOB_ID.$TASK_ID.out
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers PI05_SERVER_TIMEOUT_SEC=600
cd ${SE_VLA_ROOT}
M=$(printf '${WORK_ROOT}/lora_dagger/diag_stage2base/batch/manifest_%02d.json' "$SGE_TASK_ID")
eval "$(python3 - "$M" <<'PY'
import json,shlex,sys
m=json.load(open(sys.argv[1]))
for k in ['output_dir','task_suite_name','task_level','task_id','offset','seed','replan_steps','trials','arm']:
    v=m.get(k); print(f'{k}={shlex.quote("" if v is None else str(v))}')
PY
)"
mkdir -p "$output_dir"; export PI05_SERVER_LOG_DIR="$output_dir" SE_VLA_SRD_TELEMETRY_PATH="$output_dir/srd.jsonl"
# PURE BASE on the stage2 path: aegis source OFF
export SE_VLA_AEGIS_SOURCE=0 SE_VLA_SRD_ENABLED=1 SE_VLA_SRD_WRITES_ENABLED=0
export SE_VLA_SRD_NEAR_DISTANCE_M=0.205 SE_VLA_SRD_RELEASE_DISTANCE_M=0.307 SE_VLA_SRD_MIN_CLOSING_M_PER_STEP=0.0027
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_stage2/run.py --output "$output_dir/result.json" --task-suite "$task_suite_name" \
  --task-level "$task_level" --task-id "$task_id" --offset "$offset" --seed "$seed" \
  --replan-steps "$replan_steps" --trials "$trials" --arm "$arm" --port-base 28000 \
  --array-task-id "$SGE_TASK_ID" --server-attempts 3
