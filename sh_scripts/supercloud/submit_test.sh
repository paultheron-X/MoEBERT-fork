#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name moebert_test_bash
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=21-00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=paulth@mit.edu
#SBATCH --output=/home/gridsan/ptheron/MoEBERT-fork/logs/experiments_test_out%j.txt
#SBATCH --error=/home/gridsan/ptheron/MoEBERT-fork/logs/experiments_test_err%j.txt

echo "Launching seeds finetuning for dataset rte"

# Initialize the module command first
source /etc/profile

# Load modules
module load anaconda/2021b

# Call your script as you would from your command line
source activate MoEBERT

bash sh_scripts/test.sh