echo "Launching larger jobs for $1"

if [ $1 = "qnli" ]
then
    export eval_steps=1000
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 19 --exp 101)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 32 --exp 102)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 28 --exp 103)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 65 --exp 104) # Experimental
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 88 --exp 105) # Experimental
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
elif [ $1 = "qqp" ]
then
    export eval_steps=2000
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 10 --exp 101)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 23 --exp 102)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 35 --exp 103)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 49 --exp 104)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 88 --exp 105)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
elif [ $1 = "sst2" ]
then
    export eval_steps=1000
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 14 --exp 101)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 3 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 37 --exp 102)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 3 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 92 --exp 103)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 3 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 50 --exp 104)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 3 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 63 --exp 105)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 3 True
elif [ $1 = "mnli" ]
then
    export eval_steps=2000
    # bash sh_scripts/experiments/base_moebert_trainer.sh $1 $name $bs $weight_decay $lr $entropy $gamma $distill $eval_steps $best_epoch_first_training
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 2 --exp 101)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 4 --exp 102)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 44 --exp 103)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 64 --exp 104)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 100 --exp 105)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
elif [ $1 = "rte" ]
then
    # bash sh_scripts/experiments/base_moebert_trainer.sh $1 $name $bs $weight_decay $lr $entropy $gamma $distill $eval_steps $best_epoch_first_training $seed
    export eval_steps=1000
    declare -a StringArray=("3" "20" "35" "71" "94")
    for ((i=0; i<${#StringArray[@]}; i++)); do
    # Access the current element of the array using the index variable
        element=${StringArray[$i]}
        j=$((i+1))
        for l in {1..5}
        do
            args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n $element --exp 10$j$l)
            bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True $l
        done
    done    
elif [ $1 = "cola" ]
then
    export eval_steps=1000
    declare -a StringArray=("3" "39" "43" "36" "6")
    for ((i=0; i<${#StringArray[@]}; i++)); do
    # Access the current element of the array using the index variable
        element=${StringArray[$i]}
        j=$((i+1))
        for l in {1..5}
        do
            args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n $element --exp 10$j$l)
            bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True $l
        done
    done    
elif [ $1 = "mrpc" ]
then
    export eval_steps=1000
    declare -a StringArray=("7" "11" "35" "56" "102")
    for ((i=0; i<${#StringArray[@]}; i++)); do
    # Access the current element of the array using the index variable
        element=${StringArray[$i]}
        j=$((i+1))
        for l in {1..5}
        do
            args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n $element --exp 10$j$l)
            bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True $l
        done
    done    
fi