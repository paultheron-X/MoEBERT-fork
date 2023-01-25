#!/bin/bash

echo "Launching larger jobs for $1"
echo "Output dir is $2"
echo "Experiment Set is $3"

if [ -z $3 ]
then
    echo "No experiment set passed"
    export experiment_set=1
else
    echo "Experiment set passed is $3"
    export experiment_set=$3
fi

if [ $1 = "qnli" ]
then
    bash sh_scripts/experiments/hyper_param_hash_perm_metatuner.sh qnli $3 2e-5 32 0 7 $2
elif [ $1 = "qqp" ]
then
    bash sh_scripts/experiments/hyper_param_hash_perm_metatuner.sh qqp $3 2e-5 16 0.1 2 $2
elif [ $1 = "sst2" ]
then
    bash sh_scripts/experiments/hyper_param_hash_perm_metatuner.sh sst2 $3 2e-5 16 0 3 $2
elif [ $1 = "mnli" ]
then
    bash sh_scripts/experiments/hyper_param_hash_perm_metatuner.sh mnli $3 3e-5 16 0.01 2 $2
elif [ $1 = "rte" ]
then
    bash sh_scripts/experiments/hyper_param_hash_perm_metatuner.sh rte $3 2e-5 8 0.01 6 $2
elif [ $1 = "cola" ]
then
    bash sh_scripts/experiments/hyper_param_hash_perm_metatuner.sh cola $3 3e-5 32 0.01 7 $2
elif [ $1 = "mrpc" ]
then
    bash sh_scripts/experiments/hyper_param_hash_perm_metatuner.sh mrpc $3 4e-5 8 0.1 6 $2
fi