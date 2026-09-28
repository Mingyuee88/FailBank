#!/usr/bin/env bash
#$ -N glm45v
#$ -cwd
#$ -q gpu@@GROUP
#$ -l gpu_card=4
#$ -l h=HOST_B
#$ -pe smp 16
#$ -o ${WORK_ROOT}/glm45v_serve/serve.$JOB_ID.out
#$ -e ${WORK_ROOT}/glm45v_serve/serve.$JOB_ID.out
#
# Serve GLM-4.5V locally -- the VLM the published AEGIS calls (utils.py:475
# model="glm-4.5v"). Replaces a metered, drifting hosted endpoint with a pinned checkpoint.
#
# HOST CHOICE, and the conflict it creates. GLM-4.5V is 106B MoE. There is no official FP8
# build of the VISION variant -- zai-org publishes FP8 for GLM-4.5 and GLM-4.5-Air, both
# text-only -- so BF16 (~212 GB) must be quantized at load, which needs native FP8. Of the
# hostgroup only the L40S nodes are Ada: the a6k nodes are Ampere with no FP8 path and only
# 3x48=144 GB, which BF16 does not fit, and the 2080ti nodes are 11 GB cards.
#
# So the model can only be served on a pinned comparison host. That is a real cost: while
# this runs, HOST_B cannot host transfer-domain evaluation, and GPU model alone moves
# policy-induced cost 17.5% and flips about one cell in twelve, so those cells cannot
# simply be run elsewhere. The server is therefore scheduled in windows, not left up:
#   - concordance check and dev-stratum work: serve here on 005, evaluate on 004;
#   - transfer-domain work: take the server down first.
# If that proves too disruptive the fallback is GLM-4.1V-9B-Thinking on an a6k node, which
# fits in BF16 on a single card and frees both L40S -- but it is a different model from the
# one the paper calls, so it may only be adopted on evidence from the concordance check,
# never for convenience.
#
# Set SE_VLA_GLM_QUANT to override the precision without editing this file.
set -euo pipefail
ROOT=${SE_VLA_ROOT}/external/glm_serve
WEIGHTS=${WORK_ROOT}/glm45v
OUT=${WORK_ROOT}/glm45v_serve
mkdir -p "$OUT"

# vLLM JIT-compiles kernels at engine init and needs nvcc; compute nodes have no
# /usr/local/cuda, so torch.compile dies with "Could not find nvcc". The module supplies
# 13.2, matching this venv's torch (cu130). --enforce-eager is the second line of defence:
# it skips CUDA-graph capture entirely, throughput we do not need -- the shield makes
# exactly one VLM call per episode.
module load cuda/13.2.1 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-/software/c/cuda/13.2.1}
export PATH="$CUDA_HOME/bin:$PATH"
# vLLM shells out to ninja for JIT builds; the script execs .venv/bin/vllm directly rather
# than activating the venv, so the venv bin dir must be on PATH explicitly or ninja is
# invisible to the worker subprocesses.
export PATH="${SE_VLA_ROOT}/external/glm_serve/.venv/bin:$PATH"
export HF_HOME=${WORK_ROOT}/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
PORT=${SE_VLA_GLM_PORT:-8501}

hostname > "$OUT/host.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> "$OUT/host.txt"
echo "=== serving GLM-4.5V on $(hostname):$PORT ==="
cat "$OUT/host.txt"

# The served name must be exactly what utils.py asks for, so their call site needs no edit.
exec "$ROOT/.venv/bin/vllm" serve "$WEIGHTS" \
  --served-model-name glm-4.5v \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image":1}' \
  --reasoning-parser glm45 \
  --enforce-eager \
  --gpu-memory-utilization 0.92 \
  --quantization "${SE_VLA_GLM_QUANT:-fp8}"
