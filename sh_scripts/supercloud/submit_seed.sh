#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:volta:1
#SBATCH --time=21-00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=paulth@mit.edu
#SBATCH --output=gwnet_%j.out
#SBATCH --error=gwnet_%j.err

# Initialize the module command first
source /etc/profile

# Load modules
module load anaconda/2021a
#module load gurobi/gurobi-903

# Call your script as you would from your command line
source activate MoEBERT

export HDF5_USE_FILE_LOCKING=FALSE

cd /home/gridsan/ptheron/MoEBERT-fork

bash sh_scripts/experiments/launch_more_seeds.sh rte 