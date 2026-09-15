#!/bin/bash
#SBATCH --job-name=cm_varsel
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=12g
#SBATCH --output=logs/varsel_%A_%a.out
#SBATCH --error=logs/varsel_%A_%a.err
mkdir -p logs
# module load r                     # site-specific modules, edit as needed
# export R_LIBS_USER=...            # site-specific library path, edit as needed
export OMP_NUM_THREADS=1
export NCORES=$SLURM_CPUS_PER_TASK NPERM=2000 THR=0.2 NBIN=5 CHUNK=50
export CFG=${CFG:-6}
cd "$(dirname "$0")/.."
Rscript cm_varsel.R $CFG $SLURM_ARRAY_TASK_ID
