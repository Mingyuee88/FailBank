#!/usr/bin/env bash
# E14 login-node watcher: (1) release held pi0.5 L1 r1 base arrays once GLM is running;
# (2) when the 4 AEGIS smoke cells finish, run the validity gate; on pass submit the full AEGIS arrays.
S=${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913
O=${WORK_ROOT}/lora_dagger/review0913; J=${SE_VLA_ROOT}/joblogs
IDS=$O/lists/job_ids.txt; GLM=$1; HELD="$2"; SMOKEJOBS="$3"
SMOKE="$O/e14/l1_t0/aegis/off0_r0 $O/e14/l1_t1/aegis/off0_r0 $O/e14/pi0_l1_t0/aegis/off0_r0 $O/e14/pi0_l1_t1/aegis/off0_r0"
released=0
for i in $(seq 1 480); do
  if [ $released = 0 ] && qstat -u USER | awk -v j=$GLM '$1==j{print $5}' | grep -q r; then
    cd ${SE_VLA_ROOT}; o=$(qsub -terse -q long -o $J $S/glm_probe.sh); P=${o%%.*}; echo "E14_glm_probe2 $P" | tee -a $IDS
    for j in $SMOKEJOBS; do qalter -hold_jid $P $j >/dev/null; done
    qrls $HELD && echo "$(date +%m-%d_%H:%M) GLM running; probe $P; smoke re-held on probe; released $HELD"; released=1; fi
  n=0; for d in $SMOKE; do [ -s $d/result.json ] && n=$((n+1)); done
  if [ $n = 4 ]; then
    python3 $S/e14_aegis_gate.py $SMOKE | tee $O/lists/e14_aegis_gate.txt
    if tail -1 $O/lists/e14_aegis_gate.txt | grep -q GATE_PASS; then
      cd ${SE_VLA_ROOT}
      SU=safety_hazard_avoidance
      for t in 0 1 2 3 4; do
        o=$(qsub -terse -N RvE14a -q gpu@HOST_E -t 1-100 -tc 1 -o $J -v ARM=aegis,CK=-,SUITE=$SU,LEVEL=1,TID=$t,NOFF=50,REPS=2,OUTROOT=$O/e14/l1_t$t,EXPECT_HOST=HOST_E,PORTBASE=$((44000 + 2100 * t)),SHIELD=aegis,EXPORT_SUITE=1 $S/eval.sh); echo "E14_aegis05_l1t$t ${o%%.*}" | tee -a $IDS
        o=$(qsub -terse -N RvE14a -q gpu@HOST_D -t 1-150 -tc 1 -o $J -v ARM=aegis,CK=-,SUITE=$SU,LEVEL=2,TID=$t,NOFF=50,REPS=3,OUTROOT=$O/e14/l2_t$t,EXPECT_HOST=HOST_D,PORTBASE=$((10000 + 3100 * t)),SHIELD=aegis,EXPORT_SUITE=1 $S/eval.sh); echo "E14_aegis05_l2t$t ${o%%.*}" | tee -a $IDS
        o=$(qsub -terse -N RvE14a -q gpu@HOST_A -t 1-50 -tc 1 -o $J -v ARM=aegis,CK=-,SUITE=$SU,LEVEL=1,TID=$t,NOFF=50,REPS=1,OUTROOT=$O/e14/pi0_l1_t$t,EXPECT_HOST=HOST_A,PORTBASE=$((30000 + 1000 * t)),SHIELD=aegis,EXPORT_SUITE=1,PI0=1 $S/eval.sh); echo "E14_aegis0_l1t$t ${o%%.*}" | tee -a $IDS
        o=$(qsub -terse -N RvE14a -q gpu@HOST_C -t 1-150 -tc 1 -o $J -v ARM=aegis,CK=-,SUITE=$SU,LEVEL=2,TID=$t,NOFF=50,REPS=3,OUTROOT=$O/e14/pi0_l2_t$t,EXPECT_HOST=HOST_C,PORTBASE=$((40000 + 3100 * t)),SHIELD=aegis,EXPORT_SUITE=1,PI0=1 $S/eval.sh); echo "E14_aegis0_l2t$t ${o%%.*}" | tee -a $IDS
      done
      echo "AEGIS_FULL_SUBMITTED"
    else echo "AEGIS_NOT_SUBMITTED gate failed"; fi
    [ $released = 1 ] || { qrls $HELD; echo "released $HELD at end"; }
    echo WATCH_DONE; exit 0
  fi
  sleep 180
done
qrls $HELD; echo WATCH_TIMEOUT
