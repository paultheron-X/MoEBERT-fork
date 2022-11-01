export num_gpus=3
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="/home/paultheron/MoEBERT-fork/results/"
echo "Script name is: $0"
echo "Task name is $1"

export original_model_dir=$output_dir/"experiment_$1_finetuned_model/model"

if [ $1 = 'cola' ] || [ $1 = 'rte' ] || [ $1 = 'mrpc' ]
then
    python examples/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name $1 \
    --preprocess_importance \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --logging_steps 20 \
    --report_to tensorboard \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --warmup_ratio 0.0 \
    --seed 0 \
    --weight_decay 0.0 \
    --fp16 
else
    python examples/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name $1 \
    --preprocess_importance \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --num_train_epochs 10 \
    --logging_steps 20 \
    --report_to tensorboard \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_strategy epoch \
    --warmup_ratio 0.0 \
    --seed 0 \
    --weight_decay 0.0 \
    --fp16 
fi