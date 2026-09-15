#!/bin/bash
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem=8g
#SBATCH --output=logs/multi_%A_%a.out
#SBATCH --error=logs/multi_%A_%a.err
# cm_multi.sh -- run RPT consecutive replications of one cm_*.R script per array
# task (reps (t-1)*RPT+1 .. t*RPT), useful when a cluster caps the number of
# queued array tasks. env: SCRIPT (cm_est.R | cm_est_rl.R | cm_est_xgb.R |
# cm_test.R), PREFIX (output file prefix inside OUT_DIR: est | est_rl |
# test_<TAU_TYPE>), CFG (1-8), RPT, OUT_DIR, plus whatever the script reads
# (TAU_TYPE, CM_RHO, CM_GAUSS, NPERM). Reps whose output file exists are
# skipped, so a timed-out task is finished by resubmitting the same array;
# the task exits 1 if any replication failed.
#   sbatch --job-name=r50_tau3_c6 --array=1-50 --export=ALL,SCRIPT=cm_test.R,PREFIX=test_tau3,OUT_DIR=out/test_rho50,TAU_TYPE=tau3,CM_RHO=0.5,CFG=6,RPT=10 slurm/cm_multi.sh
mkdir -p logs
export OMP_NUM_THREADS=1
: "${SCRIPT:?}" "${PREFIX:?}" "${OUT_DIR:?}"; export CFG=${CFG:-6} RPT=${RPT:-10}
first=$(( (SLURM_ARRAY_TASK_ID - 1) * RPT + 1 )); last=$(( SLURM_ARRAY_TASK_ID * RPT ))
echo "SCRIPT=$SCRIPT CFG=$CFG TAU_TYPE=${TAU_TYPE:-S2} CM_RHO=${CM_RHO:-0.3} CM_GAUSS=${CM_GAUSS:-0} OUT_DIR=$OUT_DIR reps $first-$last"
fail=0
for r in $(seq $first $last); do
  f=$OUT_DIR/${PREFIX}_c${CFG}_r${r}.RData
  if [ -s "$f" ]; then echo "rep $r exists -- skip"; continue; fi
  if ! Rscript $SCRIPT $CFG $r; then echo "rep $r FAILED"; fail=1; fi
done
exit $fail
