export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="/home/paultheron/MoEBERT-fork/results_moebert/"

echo "Script name is: $0"
echo "Task name is $1"
echo "experiment name is $2"
echo "Batch size is $3"
echo "weight decay is $4"
echo "learning rate is $5"
echo "eval steps is $6"
echo "gamma is $7"
echo "entropy is $8"

export saving_dir=$output_dir/"experiment_$2" # Must correspond to the line in the excel hyperparameter tuning file

python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path "../../paultheron/results/model_bert_full" \
--task_name mnli \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 8 \
--learning_rate 5e-5 \
--num_train_epochs 5 \
--output_dir $output_dir/model_bert_full_moe \
--overwrite_output_dir \
--logging_steps 20 \
--logging_dir $output_dir/log \
--report_to tensorboard \
--evaluation_strategy steps \
--eval_steps 2000 \
--save_strategy no \
--warmup_ratio 0.0 \
--seed 0 \
--weight_decay 0.0 \
--fp16 \
--moebert moe \
--moebert_distill 5.0 \
--moebert_expert_num 4 \
--moebert_expert_dim 768 \
--moebert_expert_dropout 0.1 \
--moebert_load_balance 0.0 \
--moebert_load_importance importance_$1.pkl \
--moebert_route_method hash-random \
--moebert_share_importance 512 \
