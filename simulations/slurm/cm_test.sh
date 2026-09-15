#!/bin/bash
#SBATCH --job-name=cm_test
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=6g
#SBATCH --output=logs/test_%A_%a.out
#SBATCH --error=logs/test_%A_%a.err
mkdir -p logs
# module load r                     # site-specific modules, edit as needed
# export R_LIBS_USER=...            # site-specific library path, edit as needed
export OMP_NUM_THREADS=1
export TAU_TYPE=${TAU_TYPE:-S2} CFG=${CFG:-6}
cd "$(dirname "$0")/.."
Rscript cm_test.R $CFG $SLURM_ARRAY_TASK_ID
