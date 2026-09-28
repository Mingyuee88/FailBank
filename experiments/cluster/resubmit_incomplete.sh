#!/usr/bin/env bash
# Re-queue any evaluation arm/domain that is short of its full cell count.
#
# Transport crashes are sporadic (the arena's websocket dies under load) and the resume
# rule now deletes the crashed cell so it can be retried -- but an array that has already
# walked past that index will not come back to it on its own. This sweeps for holes and
# resubmits exactly the arrays that have them. Cells that already passed are skipped by
# the job itself, so a resubmission costs nothing when there is nothing to do.
set -euo pipefail
LR=${WORK_ROOT}/lora_dagger
cd ${SE_VLA_ROOT}

declare -A NCELL=( [dev]=47 [xfer]=150 [hazard]=100 [l2]=100 )
declare -A HOST=( [dev]=HOST_A [xfer]=HOST_B \
                  [hazard]=HOST_B [l2]=HOST_E )
declare -A TC=( [dev]=4 [xfer]=2 [hazard]=2 [l2]=3 )

for arm in "$@"; do
  for dom in dev xfer hazard l2; do
    d=$LR/v32_eval/$arm/$dom
    [ -d "$d" ] || continue
    # count only cells that actually passed; a crashed cell has already been removed by
    # the job, but count defensively in case this runs while one is mid-flight
    n=$(find "$d" -name result.json 2>/dev/null | wc -l)
    want=${NCELL[$dom]}
    if [ "$n" -lt "$want" ]; then
      echo "RESUBMIT $arm/$dom  $n/$want"
      qsub -t 1-"$want" -tc "${TC[$dom]}" -l h="${HOST[$dom]}" \
           -v ARM="$arm",DOMAIN="$dom" roundG/pi05_stage2/lora/run_v32_eval.sh
    else
      echo "complete $arm/$dom  $n/$want"
    fi
  done
done
