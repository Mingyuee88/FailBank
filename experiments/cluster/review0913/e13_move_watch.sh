#!/usr/bin/env bash
# Runs on the login node (compute nodes are not SGE submit hosts). Waits for the 12-cell host check,
# then runs move_apple_lemon_004.sh once. Light: one python call every 3 minutes, max 6 hours.
R=${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913
for i in $(seq 1 120); do
  last=$(python3 $R/hostcheck_compare.py 2>&1 | tail -1)
  echo "$(date +%m-%d_%H:%M) $last"
  case "$last" in
    HOSTCHECK_PASS|HOSTCHECK_FAIL)
      echo "--- move script"; bash $R/move_apple_lemon_004.sh 2>&1
      echo "--- train verdicts"; grep -ahE "TRAIN_VERDICT" ${SE_VLA_ROOT}/joblogs/RvE13Tr.o1448944.* 2>/dev/null
      echo "WATCH_DONE"; exit 0 ;;
  esac
  sleep 180
done
echo "WATCH_TIMEOUT"
