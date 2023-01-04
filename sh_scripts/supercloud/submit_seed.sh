#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:volta:2
#SBATCH --time=21-00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=paulth@mit.edu
#SBATCH --output=/home/gridsan/ptheron/MoEBERT-fork/logs/experiments_seeds_out%j.txt
#SBATCH --error=/home/gridsan/ptheron/MoEBERT-fork/logs/experiments_seeds_err%j.txt

# Initialize the module command first
source /etc/profile

# Load modules
module load anaconda/2021a
#module load gurobi/gurobi-903

# Call your script as you would from your command line
source activate MoEBERT

if [ ! -e /proc/$(pidof nvidia-smi) ]
then
	echo "nvidia-smi does not seem to be running. exiting job"
    exit 1
fi

HF_USER_DIR="/home/gridsan/$(whoami)/.cache/huggingface"
HF_LOCAL_DIR="/state/partition1/user/$(whoami)/cache/huggingface"
mkdir -p $HF_LOCAL_DIR
rsync -a --ignore-existing $HF_USER_DIR/ ${HF_LOCAL_DIR}
export HF_HOME=${HF_LOCAL_DIR}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED="true"

export BACKEND="pytorch"
export HF_MODEL_NAME=$1

export HDF5_USE_FILE_LOCKING=FALSE

cd /home/gridsan/$(whoami)/MoEBERT-fork

bash sh_scripts/experiments/launch_more_seeds.sh rte 