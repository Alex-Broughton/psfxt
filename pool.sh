#!/bin/bash
  
##SBATCH --partition=rubin
#SBATCH --job-name=psfxt
#SBATCH --output=/sdf/home/a/abrought/simulated/psfxt/output/out-%a.txt
#SBATCH --error=/sdf/home/a/abrought/simulated/psfxt/output/err-%a.txt
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00

echo "Starting sim #" $SLURM_ARRAY_TASK_ID

source /sdf/group/rubin/sw/tag/w_2023_27/loadLSST.bash
setup lsst_distrib

echo "Loaded rubin/sw/tag/w_2023_27/loadLSST.bash"

cd /sdf/home/a/abrought/simulated/psfxt

python run.py $SLURM_ARRAY_TASK_ID

# END
