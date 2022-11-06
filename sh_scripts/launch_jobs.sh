echo "Launching jobs for $1"
echo "Launching experiment $2"

if [ $1 = 'cola' ]
then
    bash sh_scripts/hyper_param_distil_metatuner.sh cola 1 3e-5 32 0.01 7
    bash sh_scripts/hyper_param_distil_metatuner.sh cola 2 3e-5 32 0.01 7
    bash sh_scripts/hyper_param_distil_metatuner.sh cola 3 3e-5 32 0.01 7
    bash sh_scripts/hyper_param_distil_metatuner.sh cola 4 3e-5 32 0.01 7
    bash sh_scripts/hyper_param_distil_metatuner.sh cola 5 3e-5 32 0.01 7
elif [ $1 = 'rte' ]
then
    bash sh_scripts/hyper_param_distil_metatuner.sh rte 1 3e-5 16 0 2
    bash sh_scripts/hyper_param_distil_metatuner.sh rte 2 3e-5 16 0 2
    bash sh_scripts/hyper_param_distil_metatuner.sh rte 3 3e-5 16 0 2
    bash sh_scripts/hyper_param_distil_metatuner.sh rte 4 3e-5 16 0 2
    bash sh_scripts/hyper_param_distil_metatuner.sh rte 5 3e-5 16 0 2
elif [ $1 = 'mrpc' ]
then
    bash sh_scripts/hyper_param_distil_metatuner.sh mrpc $2 4e-5 8 0 2
elif [ $1 = 'sst2' ]
then
    bash sh_scripts/hyper_param_distil_metatuner.sh sst2 $2  3e-5 32 0.01 3
elif [ $1 = 'qqp' ]
then
    bash sh_scripts/hyper_param_distil_metatuner.sh qqp $2 2e-5 8 0.01 2
elif [ $1 = 'mnli' ]
then
    bash sh_scripts/hyper_param_distil_metatuner.sh mnli $2 3e-5 16 0.01 2
elif [ $1 = 'qnli' ]
then
    bash sh_scripts/hyper_param_distil_metatuner.sh qnli $2 3e-5 16 0 9
fi

