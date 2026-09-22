#!/bin/bash
# submit_v9.sh -- 2026-09-22. Pilot of the data-driven setting selection (cm_test_v9.R) in four cells whose revised kernel
# test sits near 0.07 with the original settings: rho 0 (2000,40,1) [.081, B64 helped], rho 0 (2000,40,3) [.064, B64 hurt],
# rho 0.3 (2000,20,3) [.077, B64 hurt badly], rho 0.3 (2000,40,3) [.069, B64 hurt]. tau0 only, replications 1-200 (same data
# seeds as the original / O / B64 runs). Grid l1 {1e-7,1e-5,1e-3} x batch {64,128,256} x epochs {120,300}, 5-network pilots,
# selection by out-of-bag gain. About 6-8x the original time per replication -> RPT 2, 100 tasks per cell, 12 h.
cd /nas/longleaf/home/shuaiy/project/corrmix_design; O=$PWD/out; LOG=logs/submit_v9.log
sub() { CFG=$1; RHO=$2; TAU=$3; tag=$(echo $RHO | tr -d .); [ "$tag" = "0" ] && tag=00
  name=cmT_${TAU}_r${tag}_c$CFG
  out=$(sbatch --job-name=$name --nice=0 --time=12:00:00 --mem=8g --array=1-100 --export=ALL,SCRIPT=cm_test_v9.R,PREFIX=test_$TAU,OUT_DIR=$O/v9tune_rho${tag},TAU_TYPE=$TAU,CM_RHO=$RHO,CM_K=5,CM_CLIP=0.05,CM_TUNE=1,CM_TUNE_NENS=5,BKERN=2000,CFG=$CFG,RPT=2,MKL_NUM_THREADS=1 cm_multi.sh 2>&1)
  echo "$(date) $name: $out" | tee -a $LOG; }
sub 4 0 tau0; sub 8 0 tau0; sub 6 0.3 tau0; sub 8 0.3 tau0
