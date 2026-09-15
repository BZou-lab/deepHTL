#!/bin/bash
# launch_cm.sh -- submit the correlated-mixed Scenario II grid in the requested
# order: type I error (tau0, tau3) -> screen (paper's cell, cfg 6) -> power (S2)
# -> estimation. Later stages get a larger --nice so the scheduler prefers the
# earlier ones when the association's cpu/mem caps bind. 500 reps everywhere.
#   ./launch_cm.sh            # everything
#   ./launch_cm.sh screen_all # add the screen on the other 7 cells (expensive)
set -e
cd "$(dirname "$0")/.."
mkdir -p logs out/test out/est out/varsel
mem_for() { if [ $(( $1 % 2 )) -eq 0 ]; then echo 6g; else echo 4g; fi; }   # even cfg = n 2000
if [ "${1:-all}" = "screen_all" ]; then
  for CFG in 1 2 3 4 5 7 8; do
    sbatch --export=ALL,CFG=$CFG --job-name=cm_vs_c$CFG --nice=100 --array=1-500%60 slurm/cm_varsel.sh
  done; exit 0
fi
for TAU in tau0 tau3; do for CFG in 1 2 3 4 5 6 7 8; do
  sbatch --export=ALL,CFG=$CFG,TAU_TYPE=$TAU --job-name=cm_${TAU}_c$CFG --mem=$(mem_for $CFG) --nice=0 --array=1-500%120 slurm/cm_test.sh
done; done
sbatch --export=ALL,CFG=6 --job-name=cm_vs_c6 --nice=100 --array=1-500%100 slurm/cm_varsel.sh
for CFG in 1 2 3 4 5 6 7 8; do
  sbatch --export=ALL,CFG=$CFG,TAU_TYPE=S2 --job-name=cm_S2_c$CFG --mem=$(mem_for $CFG) --nice=200 --array=1-500%120 slurm/cm_test.sh
done
for CFG in 1 2 3 4 5 6 7 8; do
  sbatch --export=ALL,CFG=$CFG --job-name=cm_est_c$CFG --nice=300 --array=1-500%60 slurm/cm_est.sh
done
