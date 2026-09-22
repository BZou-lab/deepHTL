#!/bin/bash
# submit_v6.sh -- 2026-09-21 evening. 500 replicates (same seeds as reps 1-500 of the original runs) of every cell whose
# REVISED kernel or permutation test reached 0.065 or more over 1000 replicates (summary_typeI_1000.csv):
#   rho 0   : cfg 3 (1000,40,1), 4 (2000,40,1), 5 (1000,20,3), 8 (2000,40,3)
#   rho 0.3 : cfg 4 (2000,40,1), 6 (2000,20,3), 7 (1000,40,3), 8 (2000,40,3)
# tau0 and tau3, two arms through cm_test_v6.R (cm_multi.sh driver, skips finished reps). Usage: ./submit_v6.sh [O|B64|all]
#   B64 cmB64_* nice 0   nuisance pinned to batch 64, 300 epochs, patience 50; l1 stays the ORIGINAL rule (1e-5 at sigma 1 =
#                        cfg 1-4, 1e-3 at sigma 3 = cfg 5-8) so only the optimisation budget changes. FIRST (user: results asap).
#   O   cmO_*   nice 10  ORIGINAL nuisance settings + the new test variants (kperm, stratified kperm, linear-projection
#                        kernel test). p_davies / p_score / p_perm must reproduce the original runs exactly (pairing check).
# History: 18:19 both arms submitted with B64 at nice 10 / l1 1e-5 everywhere and O at nice 0; 18:3x B64 cancelled and
# resubmitted with this configuration, O moved to nice 10 by scontrol.
cd /nas/longleaf/home/shuaiy/project/corrmix_design || exit 1; O=$PWD/out; LOG=logs/submit_v6.log; mkdir -p logs
ARMS=${1:-all}
sub() { ARM=$1; CFG=$2; RHO=$3; TAU=$4; tag=$(echo $RHO | tr -d .); [ "$tag" = "0" ] && tag=00
  case $CFG in 2|4|6|8) RPT=5; ARR=1-100; MEM=8g; TLO=4:00:00; TLB=12:00:00;; *) RPT=10; ARR=1-50; MEM=6g; TLO=3:00:00; TLB=8:00:00;; esac
  COMMON="SCRIPT=cm_test_v6.R,PREFIX=test_$TAU,TAU_TYPE=$TAU,CM_RHO=$RHO,CM_CLIP=0.05,CM_K=5,NPERM=2000,BKERN=2000,CFG=$CFG,RPT=$RPT,MKL_NUM_THREADS=1,OMP_NUM_THREADS=1"
  if [ "$ARM" = O ]; then name=cmO_${TAU}_r${tag}_c$CFG; NICE=10; TL=$TLO; EXTRA="OUT_DIR=$O/v6orig_rho${tag}"
  else name=cmB64_${TAU}_r${tag}_c$CFG; NICE=0; TL=$TLB; EXTRA="OUT_DIR=$O/v6b64_rho${tag},CM_BATCH=64,CM_EPOCH=300,CM_PATIENCE=50"
       case $CFG in 1|2|3|4) EXTRA="$EXTRA,CM_L1=1e-5";; *) EXTRA="$EXTRA,CM_L1=1e-3";; esac; fi
  out=$(sbatch --job-name=$name --nice=$NICE --time=$TL --mem=$MEM --array=$ARR --export=ALL,$COMMON,$EXTRA cm_multi.sh 2>&1)
  echo "$(date) $name: $out" | tee -a $LOG; }
for ARM in B64 O; do
  case "$ARMS" in all|$ARM) ;; *) continue;; esac
  for TAU in tau0 tau3; do
    for CFG in 3 4 5 8; do sub $ARM $CFG 0   $TAU; done
    for CFG in 4 6 7 8; do sub $ARM $CFG 0.3 $TAU; done
  done
done
