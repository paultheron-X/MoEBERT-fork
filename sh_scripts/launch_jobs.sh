echo "Launching jobs for $1"

if [ $1 = 'cola' ]
then
    bash sh_scripts/hyper_param_distil_metatuner.sh cola 1 3e-5 32 0.01 7
elif [ $1 = 'rte' ]
then
    bash sh_scripts/hyper_param_distil_metatuner.sh rte 1 3e-5 16 0 2
elif [ $1 = 'mrpc' ]
then
    bash sh_scripts/hyper_param_distil_metatuner.sh mrpc 1 4e-5 8 0 2
elif [ $1 = 'sst2' ]
then
    bash sh_scripts/hyper_param_distil_metatuner.sh sst2 1  3e-5 32 0.01 3
elif [ $1 = 'qqp' ]
then
    bash sh_scripts/hyper_param_distil_metatuner.sh qqp 1 2e-5 8 0.01 2
elif [ $1 = 'mnli' ]
then
    bash sh_scripts/hyper_param_distil_metatuner.sh mnli 1 3e-5 16 0.01 2
elif [ $1 = 'qnli' ]
then
    bash sh_scripts/hyper_param_distil_metatuner.sh qnli 1 3e-5 16 0 9
fi

