if [ $1 = "all" ]
then 
    declare -a StringArray=("squad" "cola" "mnli" "qqp" "qnli" "rte" "sst-2" "squad" "mnli-bis" "qqp-bis")
    for ds in ${StringArray[@]};
    do
        for j in {1..40}
        do
            export dir="results_gcloud/$ds/experiment_$j"
            mkdir -p $dir
            gcloud compute scp --recurse  bert-$ds:~/MoEBERT-fork/results/experiment_$j/model/all_results.json \
                bert-$ds:~/MoEBERT-fork/results/experiment_$j/model/eval_results.json \
                bert-$ds:~/MoEBERT-fork/results/experiment_$j/model/train_results.json \
                bert-$ds:~/MoEBERT-fork/results/experiment_$j/model/trainer_state.json \
                bert-$ds:~/MoEBERT-fork/results/experiment_$j/model/config.json \
                $dir
        done
    done
    python sh_scripts/python_helpers/get_results.py 
elif [ $1 = 'allb' ]
then
    declare -a StringArray=("rte" "squad" "sst-2") #"qqp" "qnli" "qqp-bis" "qqp-bis-bis" "mnli-bis") # "mnli" "sst-2" "rte" "squad" "mnli-bis" "mnli") # "squad" "qnli-bis" "cola" "rte" "mnli-bis" "mnli" 
    for ds in ${StringArray[@]};
    do
        echo "Fetching results for $ds"
        for j in {100..110}
        do
            export dir="results_gcloud/$ds/moebert_experiment_$j"
            mkdir -p $dir
            gcloud compute scp --recurse  bert-$ds:~/MoEBERT-fork/results/moebert_experiment_$j/model/all_results.json \
                bert-$ds:~/MoEBERT-fork/results/moebert_experiment_$j/model/eval_results.json \
                bert-$ds:~/MoEBERT-fork/results/moebert_experiment_$j/model/train_results.json \
                bert-$ds:~/MoEBERT-fork/results/moebert_experiment_$j/model/trainer_state.json \
                bert-$ds:~/MoEBERT-fork/results/moebert_experiment_$j/model/config.json \
                $dir
        done
    done
    python sh_scripts/python_helpers/get_results.py --advanced
else
    echo "fetching results for $1"
    for j in {1..52}
    do
        export dir_="results_gcloud/$1/experiment_$j"
        mkdir -p $dir_
        gcloud compute scp --recurse bert-$1:~/MoEBERT-fork/results/experiment_$j/model/all_results.json \
            bert-$1:~/MoEBERT-fork/results/experiment_$j/model/eval_results.json \
            bert-$1:~/MoEBERT-fork/results/experiment_$j/model/train_results.json \
            bert-$1:~/MoEBERT-fork/results/experiment_$j/model/trainer_state.json \
            bert-$1:~/MoEBERT-fork/results/experiment_$j/model/config.json \
            $dir_
    done
fi

