#!/bin/bash
# submit_l1e3_typeI.sh -- 2026-09-20: rerun of the four tau = 3, sigma = 1 type I error cells (cfg 1-4) at rho = 0 and rho = 0.3
# with the L1 penalty forced to 1e-3 (CM_L1; the original runs used 1e-5 at sigma = 1), 500 reps per cell, same seeds as the
# original runs (paired comparison). cm_test_l1.R = cm_test.R + the CM_L1 override. 50 tasks x 10 reps per array.
cd /nas/longleaf/home/shuaiy/project/corrmix_design; mkdir -p logs; O=$PWD/out; LOG=logs/submit_l1e3_typeI.log
tl_for() { if [ $(( $1 % 2 )) -eq 0 ]; then echo 8:00:00; else echo 4:00:00; fi; }
mem_for() { if [ $(( $1 % 2 )) -eq 0 ]; then echo 6g; else echo 4g; fi; }
for RHO in 0 0.3; do tag=$(echo $RHO | tr -d .); [ "$tag" = "0" ] && tag=00
  for CFG in 1 2 3 4; do name=cmL1_tau3_r${tag}_c$CFG
    if squeue -u shuaiy -h -n $name | grep -q .; then echo "$(date) $name already queued" | tee -a $LOG; continue; fi
    out=$(sbatch --job-name=$name --time=$(tl_for $CFG) --mem=$(mem_for $CFG) --array=1-50 --export=ALL,SCRIPT=cm_test_l1.R,PREFIX=test_tau3,OUT_DIR=$O/test_l1e3_rho$tag,TAU_TYPE=tau3,CM_RHO=$RHO,CM_L1=1e-3,CFG=$CFG,RPT=10,MKL_NUM_THREADS=1 cm_multi.sh 2>&1)
    echo "$(date) $name: $out" | tee -a $LOG
  done
done
