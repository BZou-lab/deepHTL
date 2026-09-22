#!/bin/bash
# submit_ext1000_c24.sh -- 2026-09-21: replicates 501-1000 for cfg 2 and 4 (sigma 1, n 2000) with the ORIGINAL settings, after the tuned
# settings (l1 1e-7, 300 epochs, patience 50, clip 0.10) showed no improvement on replicates 1-500. cm_test_v2.R is forced back to the
# original configuration by CM_L1=1e-5 CM_EPOCH=120 CM_PATIENCE=20 CM_CLIP=0.05. Also fills replicates 101-500 of the rho = 0 power runs.
# A one-task check array recomputes replicates 1-2 of cfg 2 (tau0, rho 0) so the override can be compared with the original output.
cd /nas/longleaf/home/shuaiy/project/corrmix_design; O=$PWD/out; LOG=logs/submit_ext1000.log
ORIG="CM_L1=1e-5,CM_EPOCH=120,CM_PATIENCE=20,CM_CLIP=0.05"
sub() { out=$(sbatch --job-name=$1 --nice=0 --time=8:00:00 --mem=6g --array=$2 --export=ALL,SCRIPT=cm_test_v2.R,PREFIX=test_$3,OUT_DIR=$4,TAU_TYPE=$3,CM_RHO=$5,$ORIG,CFG=$6,RPT=$7,MKL_NUM_THREADS=1 cm_multi.sh 2>&1); echo "$(date) $1: $out" | tee -a $LOG; }
sub cmX_check_c2 1 tau0 $O/check_c2_orig 0 2 2
for RHO in 0 0.3; do tag=$(echo $RHO | tr -d .); [ "$tag" = "0" ] && tag=00
  for TAU in tau0 tau3 S2; do for CFG in 2 4; do sub cmX_${TAU}_r${tag}_c$CFG 51-100 $TAU $O/test_ext_rho$tag $RHO $CFG 10; done; done
done
for CFG in 2 4; do sub cmX_S2_r00_c${CFG}b 11-50 S2 $O/test_ext_rho00 0 $CFG 10; done
