#!/usr/bin/env bash
#$ -N RvE14Gate
#$ -cwd
#$ -pe smp 1
#$ -j y
# E14 AEGIS validity gate as an SGE job. Exit 0 releases the full AEGIS arrays; exit 100 puts this
# job in error state so every -hold_jid dependent stays held (AEGIS reported as not runnable).
S=${SE_VLA_ROOT}/roundG/pi05_stage2/lora/review0913
O=${WORK_ROOT}/lora_dagger/review0913
python3 $S/e14_aegis_gate.py $O/e14/l1_t0/aegis/off0_r0 $O/e14/l1_t1/aegis/off0_r0 $O/e14/pi0_l1_t0/aegis/off0_r0 $O/e14/pi0_l1_t1/aegis/off0_r0 | tee $O/lists/e14_aegis_gate.txt
tail -1 $O/lists/e14_aegis_gate.txt | grep -q GATE_PASS && exit 0
exit 100
