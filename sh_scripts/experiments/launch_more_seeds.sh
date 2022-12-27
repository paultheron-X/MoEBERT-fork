#!/bin/bash

echo "Launching larger jobs for $1"

if [ $1 = "qnli" ]
then
    export eval_steps=1000
    declare -a StringArray=("19" "32" "28" "65" "88")
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
elif [ $1 = "qqp" ]
then
    export eval_steps=2000
    declare -a StringArray=("10" "23" "35" "49" "88")
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
elif [ $1 = "sst2" ]
then
    export eval_steps=1000
    declare -a StringArray=("14" "37" "92" "50" "63")
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
elif [ $1 = "mnli" ]
then
    export eval_steps=2000
    declare -a StringArray=("2" "4" "44" "64" "100")
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