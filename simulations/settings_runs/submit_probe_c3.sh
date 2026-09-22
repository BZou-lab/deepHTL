#!/bin/bash
# submit_probe_c3.sh -- 2026-09-21: does the L1 penalty matter in the cell n=1000, p=40, sigma=1 (cfg 3)? rho = 0, tau0 and tau3, 500 reps,
# same seeds as the original run (l1 = 1e-5: revised kernel 0.080 / 0.080, revised perm 0.042 / 0.058). Everything else as in the original run.
cd /nas/longleaf/home/shuaiy/project/corrmix_design; O=$PWD/out; LOG=logs/submit_probe_c3.log
for L in 1e-7 1e-4 1e-3; do for TAU in tau0 tau3; do
  [ "$L" = "1e-3" ] && [ "$TAU" = "tau3" ] && continue      # already available in out/test_l1e3_rho00
  name=cmP_${TAU}_c3_l$L
  out=$(sbatch --job-name=$name --nice=0 --time=4:00:00 --mem=4g --array=1-50 --export=ALL,SCRIPT=cm_test_v2.R,PREFIX=test_$TAU,OUT_DIR=$O/probe_c3_l$L,TAU_TYPE=$TAU,CM_RHO=0,CM_L1=$L,CM_CLIP=0.05,CFG=3,RPT=10,MKL_NUM_THREADS=1 cm_multi.sh 2>&1)
  echo "$(date) $name: $out" | tee -a $LOG
done; done
