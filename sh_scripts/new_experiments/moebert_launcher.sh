#!/bin/bash 
#SBATCH --partition=xeon-g6-volta 
#SBATCH --constraint=xeon-g6
#SBATCH -t 4-00:00
#SBATCH -o /home/gridsan/shibal/MoEBERT-results/logs/MoEBERT/mnli/moebert_experiments/lasso_tuning_%A_%a.out
#SBATCH -e /home/gridsan/shibal/MoEBERT-results/logs/MoEBERT/mnli/moebert_experiments/lasso_tuning_%A_%a.err
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --array=1-15

# Initialize the module command first
source /etc/profile

# Load modules
module load anaconda/2022b

# Call your script as you would from your command line
source activate moebert

export TOTAL_GPUS=${SLURM_NTASKS}
export GPUS_PER_NODE=1

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

cd /home/gridsan/$(whoami)/projects/MoEBERT-fork

TASK_ID=$SLURM_ARRAY_TASK_ID
EXP_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"


task_names=(mnli)   #(rte mrpc sst2 qnli qqp qnli cola) # mnli not included
gates=(lasso)
weight_decays=(0 0.01 0.1)
distillations=(5 4 3 2 1)
trimmed_lasso_regs=(5.0)    #(0.0001 0.001 0.01 0.1 1.0 5.0 10.0)

task=${task_names[TASK_ID%1]}
gate=${gates[TASK_ID%1]}
weight_decay=${weight_decays[TASK_ID%3]}
distillation=${distillations[TASK_ID%5]}
trimmed_lasso_reg=${trimmed_lasso_regs[TASK_ID%1]}

if [ $task = 'squad' ] || [ $task = 'squad_v2' ]
then
    gate='topk'
fi

echo $TASK_ID

echo $EXP_ID

echo 'Task: ' $task

export output_dir="/home/gridsan/$(whoami)/MoEBERT-results"


if [ $gate = 'topk' ]
then
    export exp_name="topk_dis_${distillation}_wdec_${weight_decay}_seed_2"
    srun bash sh_scripts/new_experiments/moebert_k2_topk_shibal.sh $task $exp_name $distillation $weight_decay 
elif [ $gate = 'lasso' ]
then
    export exp_name="dis_${distillation}_wdec_${weight_decay}_trl_${trimmed_lasso_reg}_seed_2"
    srun bash sh_scripts/new_experiments/moebert_k2_shibal.sh $task $exp_name $distillation $weight_decay $trimmed_lasso_reg
fi

rm -r $output_dir/$task/new_moebert_k2_experiment_$exp_name/model/checkpoint-*
