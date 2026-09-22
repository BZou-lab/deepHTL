#!/bin/bash
# submit_batch50.sh -- 2026-09-21. Rerun of every cell whose REVISED test exceeded 0.07 at 1000 reps, with harder-trained nuisance
# networks: batch 50 (was 256), 300 epochs (was 120), patience 50 (was 20), l1 1e-5 everywhere, K = 5 and clip 0.05 unchanged.
# Rationale: n.batch is the BATCH SIZE, so 256 gave only 3-6 gradient steps per epoch (a few hundred updates in total) against about
# 28,000 in the UNOS analysis; batch 50 with 300 epochs gives 5,000-11,000 updates. Type I error (tau0, tau3) and power (S2), 500 reps,
# same seeds as the original runs. Cells: rho 0 -> cfg 3 (n1000 p40 s1), 4 (n2000 p40 s1), 8 (n2000 p40 s3);
#                                          rho 0.3 -> cfg 4, 6 (n2000 p20 s3), 8.
cd /nas/longleaf/home/shuaiy/project/corrmix_design; O=$PWD/out; LOG=logs/submit_batch50.log
sub() { CFG=$1; RHO=$2; TAU=$3; tag=$(echo $RHO | tr -d .); [ "$tag" = "0" ] && tag=00
  if [ "$CFG" = "3" ]; then RPT=10; ARR=1-50; TL=8:00:00; MEM=6g; else RPT=5; ARR=1-100; TL=10:00:00; MEM=8g; fi
  name=cmB50_${TAU}_r${tag}_c$CFG
  out=$(sbatch --job-name=$name --nice=0 --time=$TL --mem=$MEM --array=$ARR --export=ALL,SCRIPT=cm_test_v4.R,PREFIX=test_$TAU,OUT_DIR=$O/batch50_rho${tag},TAU_TYPE=$TAU,CM_RHO=$RHO,CM_K=5,CM_L1=1e-5,CM_EPOCH=300,CM_PATIENCE=50,CM_BATCH=50,CM_NENS=30,CM_CLIP=0.05,BKERN=2000,CFG=$CFG,RPT=$RPT,MKL_NUM_THREADS=1 cm_multi.sh 2>&1)
  echo "$(date) $name: $out" | tee -a $LOG; }
for TAU in tau0 tau3 S2; do
  for CFG in 3 4 8; do sub $CFG 0 $TAU; done
  for CFG in 4 6 8; do sub $CFG 0.3 $TAU; done
done
