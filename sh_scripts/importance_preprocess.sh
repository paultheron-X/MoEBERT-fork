export num_gpus=3
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="/home/paultheron/MoEBERT-fork/results/"
echo "Script name is: $0"
echo "Task name is $1"

export original_model_dir=$output_dir/"experiment_$1_finetuned_model/model"

python examples/text-classification/run_glue.py \
        --model_name_or_path results/experiment_$1_finetuned_model/model \
        --task_name $1 \
        --preprocess_importance \
        --do_eval \
        --do_predict \
        --max_seq_length 128 \
        --output_dir results/experiment_$1_finetuned_model/model \
        --overwrite_output_dir \
        --logging_steps 20 \
        --logging_dir results/experiment_$1_finetuned_model/log \
        --report_to tensorboard \
        --evaluation_strategy steps \
        --eval_steps $eval_steps \
        --save_strategy epoch \
        --load_best_model_at_end True \
        --warmup_ratio 0.0 \
        --seed 0 \
        --weight_decay 0.0 \
        --fp16