#!/bin/bash
# submit_fix_cells.sh -- 2026-09-21. Target the cells whose REVISED KERNEL test sits at 0.077-0.091 while the permutation test on the
# same fits is near 0.05: cfg 2 (n2000 p20 s1), cfg 3 (n1000 p40 s1), cfg 4 (n2000 p40 s1) at rho 0, and cfg 2, 6 (n2000 p20 s3) at rho 0.3.
# Two arms, 500 reps each, same seeds as the original runs, original DNN settings (l1 1e-5 / 1e-3 by sigma, 120 epochs, patience 20, clip 0.05):
#   arm K5  : K = 5  -> isolates the REFERENCE DISTRIBUTION (Davies p vs permutation p of the same kernel statistic)
#   arm K10 : K = 10 -> tests whether more training data per fold (smaller nuisance bias) is what matters
# cm_test_v3.R stores p_davies_*, p_kperm_* (same statistic, permutation reference), p_score_* and p_perm_* for every replicate.
cd /nas/longleaf/home/shuaiy/project/corrmix_design; O=$PWD/out; LOG=logs/submit_fix_cells.log
sub() { K=$1; CFG=$2; RHO=$3; TAU=$4; tag=$(echo $RHO | tr -d .); [ "$tag" = "0" ] && tag=00
  case $CFG in 2|4|6) TL=12:00:00; MEM=8g; RPT=5; ARR=1-100;; *) TL=8:00:00; MEM=6g; RPT=10; ARR=1-50;; esac
  [ "$K" = "10" ] && TL=16:00:00
  L1=1e-5; [ "$CFG" -ge 5 ] && L1=1e-3
  name=cmF${K}_${TAU}_r${tag}_c$CFG
  out=$(sbatch --job-name=$name --nice=5 --time=$TL --mem=$MEM --array=$ARR --export=ALL,SCRIPT=cm_test_v3.R,PREFIX=test_$TAU,OUT_DIR=$O/fix_K${K}_rho${tag},TAU_TYPE=$TAU,CM_RHO=$RHO,CM_K=$K,CM_L1=$L1,CM_EPOCH=120,CM_PATIENCE=20,CM_CLIP=0.05,BKERN=2000,CFG=$CFG,RPT=$RPT,MKL_NUM_THREADS=1 cm_multi.sh 2>&1)
  echo "$(date) $name: $out" | tee -a $LOG; }
for K in 5 10; do
  for TAU in tau0 tau3; do
    for CFG in 2 3 4; do sub $K $CFG 0 $TAU; done
    for CFG in 2 6; do sub $K $CFG 0.3 $TAU; done
  done
done
