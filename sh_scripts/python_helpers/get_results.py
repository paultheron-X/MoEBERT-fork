# This script will read the results jsons from the directory results for each dataset and each experiment and will create a csv file with the results

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


datasets_names = [
    "rte",
    "cola",
    "sst-2",
    "mrpc",
    "squad",
    "rte-true",
    "mnli",
    "mnli-bis",
    "qnli",
    "qnli-bis",
    "qqp",
    "qqp-bis",
    "qqp-bis-bis"
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
    for ind, dict_ in enumerate(train_log["log_history"]):
        try:
            score = dict_["eval_" + metric_dict[dataset_name]]
            sparsity = dict_["eval_sparsity"]
        except KeyError:
            continue
        if not spars_:
            if score > best_score:
                best_score = score
                best_epoch = dict_["epoch"]
        else:
            if score > best_score and np.abs(sparsity - 0.75) < 0.1:
                best_score = score
                best_epoch = dict_["epoch"]

    if best_score == 0:
        best_score = all_results["eval_" + metric_dict[dataset_name]]
        best_epoch = all_results["epoch"]
    return best_score, best_epoch


def generate_best_results_table(args):
    num_experiments = 52 if not args.advanced else 110
    dict_res = {
        "experiment": [i for i in range(num_experiments + 1)],
    }

    for dataset_name in datasets_names:
        # check if the directory exists
        if not os.path.exists("results/" + dataset_name):
            continue
        else:
            dict_res_dataset = {
                "experiment": [i for i in range(num_experiments + 1)],
                dataset_name + "_best_epoch": [],
                dataset_name + "_best_metric": [],
            }
            for experiment in range(num_experiments + 1):
                if args.advanced:
                    path = f"results/{dataset_name}/moebert_experiment_{experiment}/model"
                else:
                    path = f"results/{dataset_name}/experiment_{experiment}/model"
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
                best_score, best_epoch = get_best_result_json(
                    all_results, train_log, dataset_name, spars_=args.sparsity
                )
                best_score = round(best_score, 4)

                dict_res_dataset[dataset_name + "_best_epoch"].append(best_epoch)
                dict_res_dataset[dataset_name + "_best_metric"].append(best_score)

            dataset_df = pd.DataFrame(dict_res_dataset)
            if args.advanced:
                if args.sparsity:
                    dataset_df.to_csv(
                        f"results/{dataset_name}/moebert_results_aggregated_{dataset_name}_sparsity.csv",
                        index=False,
                    )
                else:
                    dataset_df.to_csv(
                        f"results/{dataset_name}/moebert_results_aggregated_{dataset_name}.csv", index=False
                    )
            else:
                dataset_df.to_csv(f"results/{dataset_name}/results_aggregated_{dataset_name}.csv", index=False)

    # aggregate results
    df_fin = pd.DataFrame(dict_res)
    for dataset_name in datasets_names:
        try:
            if args.advanced:
                if args.sparsity:
                    df_fin = df_fin.merge(
                        pd.read_csv(
                            f"results/{dataset_name}/moebert_results_aggregated_{dataset_name}_sparsity.csv"
                        ),
                        on="experiment",
                    )
                else:
                    df_fin = df_fin.merge(
                        pd.read_csv(f"results/{dataset_name}/moebert_results_aggregated_{dataset_name}.csv"),
                        on="experiment",
                    )
            else:
                df_fin = df_fin.merge(
                    pd.read_csv(f"results/{dataset_name}/results_aggregated_{dataset_name}.csv"),
                    on="experiment",
                )
        except FileNotFoundError:
            pass
    if args.advanced:
        if args.sparsity:
            df_fin.to_csv("results/moebert_results_aggregated_sparsity.csv", index=False)
        else:
            df_fin.to_csv("results/moebert_results_aggregated.csv", index=False)
    else:
        df_fin.to_csv("results/results_aggregated.csv", index=False)


def generate_best_results_summary(args):
    if not args.advanced:
        normal = pd.read_csv("results/results_aggregated.csv")
        res = pd.DataFrame(columns=["dataset", "best_epoch", "best_metric", "best_experiment"])
        for dataset_name in datasets_names:
            try:
                indice_best = normal[dataset_name + "_best_metric"].idxmax()
                best_epoch = normal[dataset_name + "_best_epoch"][indice_best]
                best_metric = normal[dataset_name + "_best_metric"][indice_best]
                best_experiment = normal["experiment"][indice_best]
                res = pd.concat(
                    [
                        res,
                        pd.DataFrame(
                            [[dataset_name, best_epoch, best_metric, best_experiment]],
                            columns=["dataset", "best_epoch", "best_metric", "best_experiment"],
                        ),
                    ],
                    ignore_index=True,
                )
            except KeyError:
                pass
        res.to_csv("results/base_results_aggregated_summary.csv", index=False)

    else:
        moebert_sparsity = pd.read_csv("results/moebert_results_aggregated_sparsity.csv")
        moebert = pd.read_csv("results/moebert_results_aggregated.csv")
        res_sparsity = pd.DataFrame(columns=["dataset", "best_epoch", "best_metric", "best_experiment"])
        res = pd.DataFrame(columns=["dataset", "best_epoch", "best_metric", "best_experiment"])
        for dataset_name in datasets_names:
            try:
                indice_best = moebert_sparsity[dataset_name + "_best_metric"].idxmax()
                best_epoch = moebert_sparsity[dataset_name + "_best_epoch"][indice_best]
                best_metric = moebert_sparsity[dataset_name + "_best_metric"][indice_best]
                best_experiment = moebert_sparsity["experiment"][indice_best]
                res_sparsity = pd.concat(
                    [
                        res_sparsity,
                        pd.DataFrame(
                            [[dataset_name, best_epoch, best_metric, best_experiment]],
                            columns=["dataset", "best_epoch", "best_metric", "best_experiment"],
                        ),
                    ],
                    ignore_index=True,
                )
            except KeyError:
                pass
            try:
                indice_best = moebert[dataset_name + "_best_metric"].idxmax()
                best_epoch = moebert[dataset_name + "_best_epoch"][indice_best]
                best_metric = moebert[dataset_name + "_best_metric"][indice_best]
                best_experiment = moebert["experiment"][indice_best]
                res = pd.concat(
                    [
                        res,
                        pd.DataFrame(
                            [[dataset_name, best_epoch, best_metric, best_experiment]],
                            columns=["dataset", "best_epoch", "best_metric", "best_experiment"],
                        ),
                    ],
                    ignore_index=True,
                )
            except KeyError:
                pass
        res_sparsity.to_csv("results/moebert_results_aggregated_sparsity_summary.csv", index=False)
        res.to_csv("results/moebert_results_aggregated_summary.csv", index=False)


def main(args):

    generate_best_results_table(args)
    #if args.aggregate:
    generate_best_results_summary(args)


if __name__ == "__main__":
    args = parse_args()

    main(args)
