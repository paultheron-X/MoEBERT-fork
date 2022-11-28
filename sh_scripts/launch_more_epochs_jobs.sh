echo "Launching larger jobs for $1"

if [ $1 = "qnli" ]
then
    export eval_steps=1000
elif [ $1 = "qqp" ]
then
    export eval_steps=2000
elif [ $1 = "sst2" ]
then
    export eval_steps=1000
elif [ $1 = "mnli" ]
then
    export eval_steps=2000
    # bash sh_scripts/base_moebert_trainer.sh $1 $name $bs $weight_decay $lr $entropy $gamma $distill $eval_steps $best_epoch_first_training
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 2 --exp 101)
    bash sh_scripts/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 4 --exp 102)
    bash sh_scripts/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 44 --exp 103)
    bash sh_scripts/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 64 --exp 104)
    bash sh_scripts/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 100 --exp 105)
    bash sh_scripts/base_moebert_trainer.sh $1 $args $eval_steps 2 True
fi