#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name bert_first_finetuning_qnli
#SBATCH --gres=gpu:volta:2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=21-00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=paulth@mit.edu
#SBATCH --output=/home/gridsan/ptheron/MoEBERT-fork/logs/experiments_seeds_out%j.txt
#SBATCH --error=/home/gridsan/ptheron/MoEBERT-fork/logs/experiments_seeds_err%j.txt

# Initialize the module command first
source /etc/profile

# Load modules
module load anaconda/2021b
#module load gurobi/gurobi-903

# Call your script as you would from your command line
source activate MoEBERT

export TOTAL_GPUS=${SLURM_NTASKS}
export GPUS_PER_NODE=2

echo "Total number of GPUs: $TOTAL_GPUS"
echo "GPUs per node: $GPUS_PER_NODE"

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

export HDF5_USE_FILE_LOCKING=FALSE

cd /home/gridsan/$(whoami)/MoEBERT-fork

export output_dir="/home/gridsan/$(whoami)/MoEBERT-fork/results"

bash sh_scripts/experiments/base_trainer.sh qnli 6 16 0.1 2e-5 1000 $output_dir
echo "-----------"
echo "Finished experiment 6"

bash sh_scripts/experiments/base_trainer.sh qnli 7 16 0.0 3e-5 1000 $output_dir
echo "-----------"
echo "Finished experiment 7"

bash sh_scripts/experiments/base_trainer.sh qnli 8 16 0.01 4e-5 1000 $output_dir  
echo "-----------"
echo "Finished experiment 8"

bash sh_scripts/experiments/base_trainer.sh qnli 16 64 0.0 4e-5 1000 $output_dir
echo "-----------"
echo "Finished experiment 16"

bash sh_scripts/experiments/base_trainer.sh qnli 19 64 0.0 3e-5 1000 $output_dir
echo "-----------"
echo "Finished experiment 19"

bash sh_scripts/experiments/base_trainer.sh qnli 28 16 0.0 4e-5 1000 $output_dir
echo "-----------"
echo "Finished experiment 28"

bash sh_scripts/experiments/base_trainer.sh qnli 30 16 0.1 2e-5 1000 $output_dir
echo "-----------"
echo "Finished experiment 30"

bash sh_scripts/experiments/base_trainer.sh qnli 34 32 0.0 2e-5 1000 $output_dir
echo "-----------"
echo "Finished experiment 34"

bash sh_scripts/experiments/base_trainer.sh qnli 44 16 0.01 3e-5 1000 $output_dir
echo "-----------"
echo "Finished experiment 44"

bash sh_scripts/experiments/base_trainer.sh qnli 51 64 0.1 3e-5 1000 $output_dir
echo "-----------"
echo "Finished experiment 51"