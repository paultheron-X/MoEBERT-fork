echo "Running script for task $1"
echo "-----------"
echo "Running script for mode $2"

# bash base_trainer.sh $1 experiment_num bs weight_decay lr

if [ $2 = 1 ]
then
    echo "Training mode 1"
    bash base_trainer.sh $1 4 8 0.0 4e-5
    echo "-----------"
    echo "Finished experiment 4"

    bash base_trainer.sh $1 5 8 0.01 1e-5
    echo "-----------"
    echo "Finished experiment 5"

    bash base_trainer.sh $1 6 16 0.1 2e-5
    echo "-----------"
    echo "Finished experiment 6"

    bash base_trainer.sh $1 7 16 0.0 3e-5
    echo "-----------"
    echo "Finished experiment 7"

    bash base_trainer.sh $1 8 16 0.01 4e-5  
    echo "-----------"
    echo "Finished experiment 8"

    bash base_trainer.sh $1 9 16 0.1 1e-5
    echo "-----------"
    echo "Finished experiment 9"

elif [ $2 = 2 ]
then
    echo "Training mode 2"
    bash base_trainer.sh $1 10 16 0.0 2e-5
    echo "-----------"
    echo "Finished experiment 10"

    bash base_trainer.sh $1 11 32 0.01 3e-5
    echo "-----------"
    echo "Finished experiment 11"

    bash base_trainer.sh $1 12 32 0.1 4e-5
    echo "-----------"
    echo "Finished experiment 12"

    bash base_trainer.sh $1 13 32 0.0 1e-5
    echo "-----------"
    echo "Finished experiment 13"

    bash base_trainer.sh $1 14 32 0.01 2e-5
    echo "-----------"
    echo "Finished experiment 14"

    bash base_trainer.sh $1 15 32 0.1 3e-5
    echo "-----------"
    echo "Finished experiment 15"
elif [ $2 = 3 ]
then
    echo "Training mode 3"
    bash base_trainer.sh $1 16 64 0.0 4e-5
    echo "-----------"
    echo "Finished experiment 16"

    bash base_trainer.sh $1 17 64 0.01 1e-5
    echo "-----------"
    echo "Finished experiment 17"

    bash base_trainer.sh $1 18 64 0.1 2e-5
    echo "-----------"
    echo "Finished experiment 18"

    bash base_trainer.sh $1 19 64 0.0 3e-5
    echo "-----------"
    echo "Finished experiment 19"

    bash base_trainer.sh $1 20 64 0.1 4e-5
    echo "-----------"
    echo "Finished experiment 20"
else
    echo "Invalid mode"
fi
