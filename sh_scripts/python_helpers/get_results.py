# This script will read the results jsons from the directory results_gcloud for each dataset and each experiment and will create a csv file with the results

import json
import os
import pandas as pd

num_experiments = 40

datasets_names = ["cola", "sst-2", "mrpc", "qqp", "mnli", "qnli", "rte", "squad", "mnli-bis", "qqp-bis"]

metric_dict = {
    "cola": "matthews_correlation",
    "sst-2": "accuracy",
    "mrpc": "f1",
    "qqp": "f1",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
    "squad": "f1",
    "mnli-bis": "accuracy",
    "qqp-bis": "f1"
}


def get_best_result(all_results, train_log):
    best_score = 0
    best_epoch = 0
    for ind, dict_ in enumerate(train_log["log_history"]):
        try:
            score = dict_["eval_" + metric_dict[dataset_name]]
        except KeyError:
            continue
        if score > best_score:
            best_score = score
            best_epoch = dict_["epoch"]
    if best_score == 0:
        best_score = all_results["eval_" + metric_dict[dataset_name]]
        best_epoch = all_results["epoch"]
    return best_score, best_epoch


dict_res = {
    "experiment": [i for i in range(num_experiments + 1)],
}

for dataset_name in datasets_names:
    # check if the directory exists
    if not os.path.exists("results_gcloud/" + dataset_name):
        continue
    else:
        dict_res_dataset = {
            "experiment": [i for i in range(num_experiments + 1)],
            dataset_name + "_best_epoch": [],
            dataset_name + "_best_metric": [],
        }
        for experiment in range(num_experiments + 1):
            try:
                with open(f"results_gcloud/{dataset_name}/experiment_{experiment}/all_results.json") as f:
                    all_results = json.load(f)
            except FileNotFoundError:
                all_results = {
                    "eval_" + metric_dict[dataset_name]: 0,
                    "epoch": 0,
                }
            try:
                with open(f"results_gcloud/{dataset_name}/experiment_{experiment}/trainer_state.json") as f:
                    train_log = json.load(f)
            except FileNotFoundError:
                train_log = {"log_history": []}
            best_score, best_epoch = get_best_result(all_results, train_log)
            best_score = round(best_score, 4)

            dict_res_dataset[dataset_name + "_best_epoch"].append(best_epoch)
            dict_res_dataset[dataset_name + "_best_metric"].append(best_score)

        dataset_df = pd.DataFrame(dict_res_dataset)
        dataset_df.to_csv(f"results_gcloud/{dataset_name}/results_aggregated_{dataset_name}.csv", index=False)

# aggregate results
df_fin = pd.DataFrame(dict_res)
for dataset_name in datasets_names:
    try:
        df_fin = df_fin.merge(
            pd.read_csv(f"results_gcloud/{dataset_name}/results_aggregated_{dataset_name}.csv"), on="experiment"
        )
    except FileNotFoundError:
        pass

df_fin.to_csv("results_gcloud/results_aggregated.csv", index=False)
