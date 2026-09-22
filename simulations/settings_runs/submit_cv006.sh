#!/bin/bash
# submit_cv006.sh -- 2026-09-21. Rerun with CROSS-VALIDATED nuisance settings (cm_test_v5.R) of every cell whose revised test
# reached 0.06 or more at 1000 replicates. For each replicate the mini-batch size, the number of epochs and the L1 penalty are
# chosen by three-fold CV of the R-loss over {64,128,256} x {120,300} x {1e-5,1e-3}, evaluated with 5 networks on half the sample.
# Excluded (all revised values below 0.06): cfg 2 at rho 0 and cfg 7 at rho 0. Type I error (tau0, tau3) and power (S2), 500 reps,
# same seeds as the original runs. K = 5 and clip 0.05 unchanged. The selected setting is stored per replicate (n_batch, n_epoch, l1).
cd /nas/longleaf/home/shuaiy/project/corrmix_design; O=$PWD/out; LOG=logs/submit_cv006.log
sub() { CFG=$1; RHO=$2; TAU=$3; tag=$(echo $RHO | tr -d .); [ "$tag" = "0" ] && tag=00
  case $CFG in 2|4|6|8) RPT=5; ARR=1-100; TL=12:00:00; MEM=8g;; *) RPT=10; ARR=1-50; TL=10:00:00; MEM=6g;; esac
  name=cmCV_${TAU}_r${tag}_c$CFG
  out=$(sbatch --job-name=$name --nice=0 --time=$TL --mem=$MEM --array=$ARR --export=ALL,SCRIPT=cm_test_v5.R,PREFIX=test_$TAU,OUT_DIR=$O/cv006_rho${tag},TAU_TYPE=$TAU,CM_RHO=$RHO,CM_K=5,CM_CLIP=0.05,CM_TUNE=1,CM_TUNE_ENS=5,CM_TUNE_FRAC=0.5,CM_PATIENCE=50,BKERN=2000,CFG=$CFG,RPT=$RPT,MKL_NUM_THREADS=1 cm_multi.sh 2>&1)
  echo "$(date) $name: $out" | tee -a $LOG; }
for TAU in tau0 tau3 S2; do
  for CFG in 1 3 4 5 6 8;       do sub $CFG 0   $TAU; done
  for CFG in 1 2 3 4 5 6 7 8;   do sub $CFG 0.3 $TAU; done
done
