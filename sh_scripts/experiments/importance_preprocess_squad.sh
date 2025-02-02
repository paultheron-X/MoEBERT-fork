#!/bin/bash

export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
echo "Script name is: $0"
echo "Task name is $1"
echo "Given output dir is $2"
export output_dir="$2/$1"
export original_model_dir='deepset/bert-base-uncased-squad2'
#export original_model_dir='results/squad_experiment_1/model'

export saving_dir=$output_dir/"importance_moebert_experiment_$1" # Must correspond to the line in the excel hyperparameter tuning file


export metric_for_best_model="f1"


if [ $1 = "squad" ]; then
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
    examples/question-answering/run_qa.py \
    --model_name_or_path $original_model_dir \
    --dataset_name squad \
    --task_name $1 \
    --preprocess_importance \
    --do_eval \
    --max_seq_length 128 \
    --num_train_epochs 10 \
    --output_dir $saving_dir/model \
    --overwrite_output_dir \
    --logging_steps 20 \
    --logging_dir $saving_dir/log \
    --report_to tensorboard \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_strategy epoch \
    --load_best_model_at_end False \
    --metric_for_best_model $metric_for_best_model \
    --warmup_ratio 0.0 \
    --seed 0 \
    --weight_decay 0.0 \
    --fp16 

elif [ $1 = "squad_v2" ]; then
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
     examples/question-answering/run_qa.py \
    --model_name_or_path $original_model_dir \
    --dataset_name squad_v2 \
    --version_2_with_negative \
    --preprocess_importance \
    --do_eval \
    --max_seq_length 384 \
    --num_train_epochs 10 \
    --output_dir $saving_dir/model \
    --overwrite_output_dir \
    --logging_steps 20 \
    --logging_dir $saving_dir/log \
    --report_to tensorboard \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_strategy epoch \
    --load_best_model_at_end False \
    --warmup_ratio 0.0 \
    --seed 0 \
    --metric_for_best_model $metric_for_best_model \
    --weight_decay 0.0 \
    --fp16 
else
    echo "Task name not recognized"
fi