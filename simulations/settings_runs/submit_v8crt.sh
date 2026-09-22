#!/bin/bash
# submit_v8crt.sh -- 2026-09-22. Pilot of the conditional-randomization reference (cm_test_v8.R, CM_SCREEN=0, ORIGINAL settings)
# for the revised kernel test. Cells (rho 0, same seeds as the stored runs): cfg 3 (1000,40,1) tau0 + tau3 200 reps each,
# cfg 4 (2000,40,1) tau0 100 reps, cfg 5 (1000,20,3) tau0 100 reps (sanity: expect no change), cfg 3 S2 100 reps (power guard).
cd /nas/longleaf/home/shuaiy/project/corrmix_design || exit 1; O=$PWD/out; LOG=logs/submit_v8crt.log; mkdir -p logs
sub() { CFG=$1; RHO=$2; TAU=$3; ARR=$4; tag=$(echo $RHO | tr -d .); [ "$tag" = "0" ] && tag=00
  case $CFG in 2|4|6|8) RPT=5; MEM=8g; TL=4:00:00;; *) RPT=10; MEM=6g; TL=3:00:00;; esac
  name=cmC_${TAU}_r${tag}_c$CFG
  out=$(sbatch --job-name=$name --nice=0 --time=$TL --mem=$MEM --array=$ARR --export=ALL,SCRIPT=cm_test_v8.R,PREFIX=test_$TAU,TAU_TYPE=$TAU,CM_RHO=$RHO,CM_CLIP=0.05,CM_K=5,NPERM=2000,BKERN=2000,CM_SCREEN=0,CFG=$CFG,RPT=$RPT,OUT_DIR=$O/v8crt_rho${tag},MKL_NUM_THREADS=1,OMP_NUM_THREADS=1 cm_multi.sh 2>&1)
  echo "$(date) $name: $out" | tee -a $LOG; }
sub 3 0 tau0 1-20; sub 3 0 tau3 1-20; sub 4 0 tau0 1-20; sub 5 0 tau0 1-10; sub 3 0 S2 1-10
