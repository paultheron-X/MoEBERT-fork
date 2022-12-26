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
    export eval_steps=1000
    # bash sh_scripts/experiments/base_moebert_trainer.sh $1 $name $bs $weight_decay $lr $entropy $gamma $distill $eval_steps $best_epoch_first_training $seed
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 3 --exp 1011)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 1
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 3 --exp 1012)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 2
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 3 --exp 1013)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 3
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 3 --exp 1014)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 4
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 3 --exp 1015)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 5

    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 20 --exp 1021)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 1
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 20 --exp 1022)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 2
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 20 --exp 1023)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 3
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 20 --exp 1024)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 4
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 20 --exp 1025)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 5

    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 35 --exp 1031)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 1
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 35 --exp 1032)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 2
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 35 --exp 1033)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 3
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 35 --exp 1034)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 4
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 35 --exp 1035)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 5

    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 71 --exp 1041)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 1
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 71 --exp 1042)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 2
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 71 --exp 1043)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 3
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 71 --exp 1044)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 4
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 71 --exp 1045)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 5

    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 94 --exp 1051)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 1
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 94 --exp 1052)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 2
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 94 --exp 1053)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 3
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 94 --exp 1054)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 4
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 94 --exp 1055)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True 5
elif [ $1 = "cola" ]
then
    export eval_steps=1000
    # bash sh_scripts/experiments/base_moebert_trainer.sh $1 $name $bs $weight_decay $lr $entropy $gamma $distill $eval_steps $best_epoch_first_training $seed
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 3 --exp 101)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 39 --exp 102)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 43 --exp 103)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 36 --exp 104)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 6 --exp 105)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
elif [ $1 = "mrpc" ]
then
    export eval_steps=1000
    # bash sh_scripts/experiments/base_moebert_trainer.sh $1 $name $bs $weight_decay $lr $entropy $gamma $distill $eval_steps $best_epoch_first_training $seed
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 7 --exp 101)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 11 --exp 102)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 35 --exp 103)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 56 --exp 104)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n 102 --exp 105)
    bash sh_scripts/experiments/base_moebert_trainer.sh $1 $args $eval_steps 2 True
fi