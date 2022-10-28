echo "Getting best params from pretraining for task $1"
echo "Best lr is $2"
echo "Best batch size is $3"
echo "Best weight decay is $4"
echo "Best epoch is $5"

if [ $1 = "mnli" ] 
then
    export eval_steps=2000
elif [ $1 = "mrpc" ]
then
    export eval_steps=200
elif [ $1 = "qnli" ]
then
    export eval_steps=1000
elif [ $1 = "qqp" ]
then
    export eval_steps=2000
elif [ $1 = "rte" ]
then
    export eval_steps=200
elif [ $1 = "sst2" ]
then
    export eval_steps=1000
elif [ $1 = "cola" ]
then
    export eval_steps=200
fi

# Check if we have a finetuned model for this task
if [ -d "/home/paultheron/MoEBERT-fork/results/experiment_finetuned_model" ]
then
    echo "Finetuned model already exists for task $1"
else
    echo "Finetuned model does not exist for task $1"
    echo "Creating finetuned model for task $1"

    bash sh_scripts/base_trainer.sh $1 "finetuned_model" $3 $4 $2 $5 $eval_steps

    export CUDA_VISIBLE_DEVICES=0
    python examples/text-classification/run_glue.py \
        --model_name_or_path results/experiment_finetuned_model/model \
        --task_name $1 \
        --preprocess_importance \
        --do_eval \
        --do_predict \
        --max_seq_length 128 \
        --output_dir results/experiment_finetuned_model/model \
        --overwrite_output_dir \
        --logging_steps 20 \
        --logging_dir results/experiment_finetuned_model/log \
        --report_to tensorboard \
        --evaluation_strategy steps \
        --eval_steps $eval_steps \
        --save_strategy epoch \
        --load_best_model_at_end True \
        --warmup_ratio 0.0 \
        --seed 0 \
        --weight_decay 0.0 \
        --fp16
    
    echo "Finetuned model created for task $1"
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi
echo "Starting training for task $1, with the given hyperparameters"

