#!/bin/bash
# submit_v2_tests.sh -- 2026-09-21 tuning round: type I error (tau0, tau3) and power (S2) for all 8 cells at rho = 0 and 0.3,
# 500 reps per cell, cm_test_v2.R (l1 by noise and n, 300 epochs / patience 50 at sigma 1 & n 2000, clip 0.10, both kernel statistics).
# Same seeds as the original runs. Cells 2 and 4 (sigma 1, n 2000) run 5 reps per task (100 tasks), the others 10 reps per task (50 tasks).
# Time limits are sized from the K = 5 runtimes of the original arrays (n 1000: <= 80 min per 10 reps, n 2000: <= 290 min per 10 reps).
cd /nas/longleaf/home/shuaiy/project/corrmix_design; mkdir -p logs; O=$PWD/out; LOG=logs/submit_v2_tests.log
for RHO in 0 0.3; do tag=$(echo $RHO | tr -d .); [ "$tag" = "0" ] && tag=00
  for TAU in tau0 tau3 S2; do for CFG in 1 2 3 4 5 6 7 8; do name=cmV2_${TAU}_r${tag}_c$CFG
    case $CFG in 2|4) RPT=5; ARR=1-100; TL=10:00:00; MEM=6g;; 6|8) RPT=10; ARR=1-50; TL=8:00:00; MEM=6g;; *) RPT=10; ARR=1-50; TL=4:00:00; MEM=4g;; esac
    if squeue -u shuaiy -h -n $name | grep -q .; then echo "$(date) $name already queued" | tee -a $LOG; continue; fi
    out=$(sbatch --job-name=$name --nice=0 --time=$TL --mem=$MEM --array=$ARR --export=ALL,SCRIPT=cm_test_v2.R,PREFIX=test_$TAU,OUT_DIR=$O/test_v2_rho$tag,TAU_TYPE=$TAU,CM_RHO=$RHO,CFG=$CFG,RPT=$RPT,MKL_NUM_THREADS=1 cm_multi.sh 2>&1)
    echo "$(date) $name: $out" | tee -a $LOG
  done; done
done
