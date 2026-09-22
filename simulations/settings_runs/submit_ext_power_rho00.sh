#!/bin/bash
# submit_ext_power_rho00.sh -- 2026-09-21: the original rho = 0 power runs (out/test_rho00, S2) have only replicates 1-100 per cell.
# To reach 1000 replicates with the ORIGINAL settings this fills replicates 101-500 (array 11-50 x 10 reps) for cfg 1,3,5,6,7,8;
# replicates 501-1000 are already covered by submit_ext1000.sh. Same script, seeds and output folder as the extension (out/test_ext_rho00).
cd /nas/longleaf/home/shuaiy/project/corrmix_design; O=$PWD/out; LOG=logs/submit_ext1000.log
for CFG in 1 3 5 6 7 8; do name=cmX_S2_r00_c${CFG}b
  case $CFG in 6|8) TL=8:00:00; MEM=6g;; *) TL=4:00:00; MEM=4g;; esac
  out=$(sbatch --job-name=$name --nice=0 --time=$TL --mem=$MEM --array=11-50 --export=ALL,SCRIPT=cm_test_v2.R,PREFIX=test_S2,OUT_DIR=$O/test_ext_rho00,TAU_TYPE=S2,CM_RHO=0,CM_CLIP=0.05,CFG=$CFG,RPT=10,MKL_NUM_THREADS=1 cm_multi.sh 2>&1)
  echo "$(date) $name: $out" | tee -a $LOG
done
