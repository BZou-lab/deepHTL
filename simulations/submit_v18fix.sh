#!/bin/bash
# submit_v18fix.sh -- 2026-09-26. FINAL rerun of the cells whose stored type I error was >= 0.065 (1000-rep original run or 500-rep v6),
# with cm_test_v18r2.R + settings_pilot5.env: hierarchical lasso screen on [X, X^2, XiXj], mu / mu* nets on the selected raw variables
# plus the selected squares / products (augmented), e net on the selected raw variables only, mu nets batch 128 / 500 epochs / patience 50,
# e net default (256 / 120 / 20). Tests recorded: product form (primary analytic) and sign form, unrevised and revised; permutation
# fold x arm and fold x arm x e_hat-quintile (primary permutation). 1000 reps per (cell, rho, tau), RPT=1, outputs on /work (quota).
# Combos (cfg: 3 (1000,40,1) 4 (2000,40,1) 5 (1000,20,3) 6 (2000,20,3) 7 (1000,40,3) 8 (2000,40,3)):
#   rho 0  : cfg 5, 3, 4, 8        rho 0.3: cfg 7, 6, 4, 8        each tau0 and tau3
# rho 0 cfg 3 reps 1-200 come from pilot 5 (out/v18pilot5_rho00, identical settings); this script submits reps 201-1000 there.
cd /nas/longleaf/home/shuaiy/project/corrmix_design || exit 1; LOG=logs/submit_v18fix.log; mkdir -p logs
W=/work/users/s/h/shuaiy/deephtl; mkdir -p $W/v18fix_rho00 $W/v18fix_rho03 || { echo "cannot create $W"; exit 1; }
SET=$(grep -v '^#' settings_pilot5.env | grep -v '^$' | tr '\n' ',' | sed 's/,$//')
sub() { RHO=$1; CFG=$2; TAU=$3; ARR=$4; DIR=$5; case $CFG in 2|4|6|8) TL=6:00:00;; *) TL=3:00:00;; esac; tag=$(echo $RHO | tr -d .); [ "$tag" = "0" ] && tag=00
  out=$(sbatch --job-name=cmF_${TAU}_r${tag}_c$CFG --nice=0 --time=$TL --mem=8g --array=$ARR \
    --export=ALL,SCRIPT=cm_test_v18r2.R,PREFIX=test_$TAU,OUT_DIR=$DIR,TAU_TYPE=$TAU,CM_RHO=$RHO,CM_K=5,CM_CLIP=0.05,NPERM=2000,CFG=$CFG,RPT=1,MKL_NUM_THREADS=1,OMP_NUM_THREADS=1,$SET cm_multi.sh 2>&1)
  echo "$(date) cmF_${TAU}_r${tag}_c$CFG array $ARR -> $DIR [$SET]: $out" | tee -a $LOG; }
# n = 1000 combos first (finish soonest), then n = 2000
for TAU in tau0 tau3; do
  sub 0 3 $TAU 201-1000 $PWD/out/v18pilot5_rho00
  sub 0 5 $TAU 1-1000 $W/v18fix_rho00
  sub 0.3 7 $TAU 1-1000 $W/v18fix_rho03
done
for TAU in tau0 tau3; do
  sub 0 4 $TAU 1-1000 $W/v18fix_rho00; sub 0 8 $TAU 1-1000 $W/v18fix_rho00
  sub 0.3 6 $TAU 1-1000 $W/v18fix_rho03; sub 0.3 4 $TAU 1-1000 $W/v18fix_rho03; sub 0.3 8 $TAU 1-1000 $W/v18fix_rho03
done
