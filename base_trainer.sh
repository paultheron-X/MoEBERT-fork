export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="/home/paultheron/MoEBERT-fork/results/"
export saving_dir=$output_dir/experiment_1  # Must correspond to the line in the excel hyperparameter tuning file

#python -m torch.distributed.launch --nproc_per_node=$num_gpus \
python examples/text-classification/run_glue.py \
--model_name_or_path bert-base-uncased \
--task_name mnli \
--per_device_train_batch_size 8 \
--weight_decay 0.0 \
--learning_rate 1e-5 \
--do_train \
--do_eval \
--max_seq_length 128 \
--num_train_epochs 7 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 20 \
--logging_dir $output_dir/log \
--report_to tensorboard \
--evaluation_strategy steps \
--eval_steps 2000 \
--save_strategy epoch \
--warmup_ratio 0.0 \
--seed 0 \
--weight_decay 0.0 \
--fp16 \