#!/bin/bash
# submit_v7pilot.sh -- 2026-09-21 night. Pilot of the within-fold nuisance pre-screen (cm_test_v7.R, CM_SCREEN=1) with the
# ORIGINAL settings otherwise, on the cells that no hyperparameter setting has fixed: rho 0 cfg 3 (1000,40,1) tau0 + tau3,
# 200 reps each, and rho 0 cfg 8 (2000,40,3) tau0, 100 reps. Same seeds as the original reps. nice 0 (ahead of B64/O).
cd /nas/longleaf/home/shuaiy/project/corrmix_design || exit 1; O=$PWD/out; LOG=logs/submit_v7pilot.log; mkdir -p logs
sub() { CFG=$1; RHO=$2; TAU=$3; ARR=$4; tag=$(echo $RHO | tr -d .); [ "$tag" = "0" ] && tag=00
  case $CFG in 2|4|6|8) RPT=5; MEM=8g; TL=4:00:00;; *) RPT=10; MEM=6g; TL=3:00:00;; esac
  name=cmS_${TAU}_r${tag}_c$CFG
  out=$(sbatch --job-name=$name --nice=0 --time=$TL --mem=$MEM --array=$ARR --export=ALL,SCRIPT=cm_test_v7.R,PREFIX=test_$TAU,TAU_TYPE=$TAU,CM_RHO=$RHO,CM_CLIP=0.05,CM_K=5,NPERM=2000,BKERN=2000,CM_SCREEN=1,CFG=$CFG,RPT=$RPT,OUT_DIR=$O/v7scr_rho${tag},MKL_NUM_THREADS=1,OMP_NUM_THREADS=1 cm_multi.sh 2>&1)
  echo "$(date) $name: $out" | tee -a $LOG; }
sub 3 0 tau0 1-20; sub 3 0 tau3 1-20; sub 8 0 tau0 1-20
