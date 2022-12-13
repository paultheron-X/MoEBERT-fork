echo "Getting best params from pretraining for task $1"
echo "Experiment Set is $2"
echo "Best lr was $3"
echo "Best batch size was $4"
echo "Best weight decay was $5"
echo "Best epoch was $6"

if [ $1 = "mnli" ] 
then
    export eval_steps=2000
elif [ $1 = "mrpc" ]
then
    export eval_steps=100
elif [ $1 = "qnli" ]
then
    export eval_steps=1000
elif [ $1 = "qqp" ]
then
    export eval_steps=2000
elif [ $1 = "rte" ]
then
    export eval_steps=100
elif [ $1 = "sst2" ]
then
    export eval_steps=1000
elif [ $1 = "cola" ]
then
    export eval_steps=100
fi


# Check if we have a finetuned model for this task
if [ -d "/home/paultheron/MoEBERT-fork/results/experiment_$1_finetuned_model/model" ]
then
    echo "Finetuned model already exists for task $1"
else
    echo "Finetuned model does not exist for task $1"
    echo "Creating finetuned model for task $1"

    bash sh_scripts/base_trainer.sh $1 "$1_finetuned_model" $4 $5 $3 $eval_steps

    echo "Finetuned model created for task $1"
    echo "Now preprocessing importance for this task"
    bash sh_scripts/importance_preprocess.sh $1

    python merge_importance.py --task $1 --num_files 3
    
    echo "Finetuned model created for task $1"

fi
echo "Starting training for task $1, with the given hyperparameters"

echo "Now starting distillation for task $1"

echo "Launching Experiment Set $2"

BEGIN=$((20*$2 + -19))
END=$((20*$2))


for i in $(eval echo "{$BEGIN..$END}")
do
    echo "Launching Moebert Experiment $i"
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n $i)

    bash sh_scripts/base_moebert_trainer.sh $1 $args $eval_steps $6 True
done
