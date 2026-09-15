#!/bin/bash
#SBATCH --job-name=cm_estrl
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=6g
#SBATCH --output=logs/estrl_%A_%a.out
#SBATCH --error=logs/estrl_%A_%a.err
mkdir -p logs
# module load r                     # site-specific modules, edit as needed
# export R_LIBS_USER=...            # site-specific library path, edit as needed
export OMP_NUM_THREADS=1
export CFG=${CFG:-6}
cd "$(dirname "$0")/.."
Rscript cm_est_rl.R $CFG $SLURM_ARRAY_TASK_ID
