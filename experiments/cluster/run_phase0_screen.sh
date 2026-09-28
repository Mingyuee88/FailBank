#!/usr/bin/env bash
#$ -N p0_base_scan
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=1
#$ -l h=(A6K_HOSTS*|L40S_HOSTS*)&!HOST_C
#$ -pe smp 2
#$ -t 1-141
#$ -tc 4
#$ -o ${WORK_ROOT}/lora_dagger/phase0_base_scan/batch/j.$JOB_ID.$TASK_ID.out
#$ -e ${WORK_ROOT}/lora_dagger/phase0_base_scan/batch/j.$JOB_ID.$TASK_ID.out
# Phase0 base scan: pure base / pass-through. Mirrors the PROVEN launcher
# stage2_srd/p0_variance/run.sh (roundG/pi05_static_sweep/run.py). NO aegis/SRD
# env vars -- SRD telemetry (step_cost + pre_step_features) is default-on here.
# set -t to 1-<n> from phase0_screen.py output (offsets 0-47 excl 9 x seeds{7,11,13}=141).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HOME=${WORK_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${WORK_ROOT}/hf_cache/transformers PI05_SERVER_TIMEOUT_SEC=600
cd ${SE_VLA_ROOT}
M=$(printf '${WORK_ROOT}/lora_dagger/phase0_base_scan/batch/manifest_%04d.json' "$SGE_TASK_ID")
eval "$(python3 - "$M" <<'PY'
import json,shlex,sys
m=json.load(open(sys.argv[1]))
for k in ['output_dir','task_suite_name','task_level','task_id','offset','seed','replan_steps','trials','arm','code_checksum']:
    v=m.get(k); print(f'{k}={shlex.quote("" if v is None else str(v))}')
PY
)"
export PI05_SERVER_LOG_DIR="$output_dir"; mkdir -p "$output_dir"
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  roundG/pi05_static_sweep/run.py --output "$output_dir/result.json" --task-suite "$task_suite_name" \
  --task-level "$task_level" --task-id "$task_id" --offset "$offset" --seed "$seed" \
  --replan-steps "$replan_steps" --trials "$trials" --arm "$arm" --port-base 26000 \
  --array-task-id "$SGE_TASK_ID" --server-attempts 3
