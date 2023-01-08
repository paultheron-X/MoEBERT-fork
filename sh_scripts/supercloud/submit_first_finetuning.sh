#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name bert_first_finetuning_$1
#SBATCH --gres=gpu:volta:2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
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

if [ -z "$2" ]
then
    echo "No output dir passed"
    export output_dir="/home/gridsan/$(whoami)/MoEBERT-fork/results"
else
    echo "Given Output dir is $2"
    export output_dir=$2
fi

bash sh_scripts/experiments/hyperparameter_metatuner.sh $1 1 $output_dir
bash sh_scripts/experiments/hyperparameter_metatuner.sh $1 2 $output_dir
bash sh_scripts/experiments/hyperparameter_metatuner.sh $1 3 $output_dir
bash sh_scripts/experiments/hyperparameter_metatuner.sh $1 7 $output_dir
