import json
import os
import pandas as pd
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="If set, the script will also aggregate the results of the advanced experiments",
    )
    parser.add_argument(
        "--sparsity",
        "-s",
        action="store_true",
        help="If set, the script will also aggregate the results of the sparsity experiments",
        default=False,
    )
    parser.add_argument(
        "--aggregate",
        "-a",
        action="store_true",
        help="If set, the script will aggregate the results of the experiments",
        default=False,
    )
    return parser.parse_args()


mnli_names = [
    "mnli",
    "mnli-bis",
    "rte",
]

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
    "qqp-bis": "f1",
    "qnli-bis": "accuracy",
    "rte-true": "accuracy",
    "qqp-bis-bis": "f1",
}


def get_best_result_json(all_results, train_log, dataset_name, spars_):
    best_score = 0
    best_epoch = 0
    score_mm_best = 0
    score_m_best = 0
    for ind, dict_ in enumerate(train_log["log_history"]):
        try:
            score = dict_["eval_" + metric_dict[dataset_name]]
            sparsity = dict_["eval_sparsity"]
            score_mm = dict_["eval_mm_" + metric_dict[dataset_name]]
            score_m = dict_["eval_m_" + metric_dict[dataset_name]]
        except KeyError:
            continue
        if not spars_:
            if score > best_score:
                best_score = score
                best_epoch = dict_["epoch"]
                score_mm_best = score_mm
                score_m_best = score_m
        else:
            if score > best_score and np.abs(sparsity - 0.75) < 0.1:
                best_score = score
                best_epoch = dict_["epoch"]
                score_mm_best = score_mm
                score_m_best = score_m

    if best_score == 0:
        best_score = all_results["eval_" + metric_dict[dataset_name]]
        best_epoch = all_results["epoch"]
    return best_score, best_epoch, score_mm_best, score_m_best


def generate_best_results_table(args):
    num_experiments = 52 if not args.advanced else 105
    dict_res = {
        "experiment": [i for i in range(num_experiments + 1)],
    }

    for dataset_name in mnli_names:
        # check if the directory exists
        if not os.path.exists("results_gcloud/" + dataset_name):
            continue
        else:
            dict_res_dataset = {
                "experiment": [i for i in range(num_experiments + 1)],
                dataset_name + "_best_epoch": [],
                dataset_name + "_best_metric": [],
                dataset_name + "_best_metric_mm": [],
                dataset_name + "_best_metric_m": [],
            }
            for experiment in range(num_experiments + 1):
                if args.advanced:
                    path = f"results_gcloud/{dataset_name}/moebert_experiment_{experiment}"
                else:
                    path = f"results_gcloud/{dataset_name}/experiment_{experiment}"
                try:
                    with open(path + "/all_results.json") as f:
                        all_results = json.load(f)
                except FileNotFoundError:
                    all_results = {
                        "eval_" + metric_dict[dataset_name]: 0,
                        "epoch": 0,
                    }
                try:
                    with open(path + "/trainer_state.json") as f:
                        train_log = json.load(f)
                except FileNotFoundError:
                    train_log = {"log_history": []}
                best_score, best_epoch , score_mm_best, score_m_best = get_best_result_json(
                    all_results, train_log, dataset_name, args.sparsity
                )
                best_score = round(best_score, 4)
                score_mm_best = round(score_mm_best, 4)
                score_m_best = round(score_m_best, 4)

                dict_res_dataset[dataset_name + "_best_epoch"].append(best_epoch)
                dict_res_dataset[dataset_name + "_best_metric"].append(best_score)
                dict_res_dataset[dataset_name + "_best_metric_mm"].append(score_mm_best)
                dict_res_dataset[dataset_name + "_best_metric_m"].append(score_m_best)
                

            dataset_df = pd.DataFrame(dict_res_dataset)
            if args.advanced:
                if args.sparsity:
                    dataset_df.to_csv(
                        f"results_gcloud/{dataset_name}/moebert_mnli_results_aggregated_{dataset_name}_sparsity.csv",
                        index=False,
                    )
                else:
                    dataset_df.to_csv(
                        f"results_gcloud/{dataset_name}/moebert_mnli_results_aggregated_{dataset_name}.csv", index=False
                    )
            else:
                dataset_df.to_csv(f"results_gcloud/{dataset_name}/results_mnli_aggregated_{dataset_name}.csv", index=False)

    # aggregate results
    df_fin = pd.DataFrame(dict_res)
    for dataset_name in mnli_names:
        try:
            if args.advanced:
                if args.sparsity:
                    df_fin = df_fin.merge(
                        pd.read_csv(
                            f"results_gcloud/{dataset_name}/moebert_mnli_results_aggregated_{dataset_name}_sparsity.csv"
                        ),
                        on="experiment",
                    )
                else:
                    df_fin = df_fin.merge(
                        pd.read_csv(f"results_gcloud/{dataset_name}/moebert_mnli_results_aggregated_{dataset_name}.csv"),
                        on="experiment",
                    )
            else:
                df_fin = df_fin.merge(
                    pd.read_csv(f"results_gcloud/{dataset_name}/results_mnli_aggregated_{dataset_name}.csv"),
                    on="experiment",
                )
        except FileNotFoundError:
            pass
    if args.advanced:
        if args.sparsity:
            df_fin.to_csv("results_gcloud/moebert_mnli_results_aggregated_sparsity.csv", index=False)
        else:
            df_fin.to_csv("results_gcloud/moebert_mnli_results_aggregated.csv", index=False)
    else:
        df_fin.to_csv("results_gcloud/results_mnli_aggregated.csv", index=False)

    

def main(args):

    generate_best_results_table(args)


if __name__ == "__main__":
    args = parse_args()

    main(args)
