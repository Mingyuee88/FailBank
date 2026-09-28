#!/usr/bin/env bash
#$ -N RvGlmProbe
#$ -cwd
#$ -pe smp 1
#$ -j y
# Review 2026-09-14, E10: gate the AEGIS arm on the GLM-4.5V endpoint. Exits 0 once
# /v1/models answers 200 and lists glm-4.5v. Exits 100 after 3 h so that SGE puts this job in
# error state and every -hold_jid dependent stays held instead of running without perception.
URL=${GLM_URL:-http://HOST_E:8502/v1/models}
for i in $(seq 1 180); do
  body=$(curl -s -m 10 "$URL" || true)
  if echo "$body" | grep -q '"glm-4.5v"'; then
    echo "GLM_READY after ${i} min: $(date)"; exit 0
  fi
  sleep 60
done
echo "GLM_NOT_READY after 180 min: $(date)"
exit 100
