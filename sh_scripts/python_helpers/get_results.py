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
    parser.add_argument(
        "--perm",
        "-p",
        action="store_true",
        help="If set, the script will aggregate the results of the experiments for permutation",
        default=False,
    )
    parser.add_argument(
        "--ktwo",
        "-k",
        action="store_true",
        help="If set, the script will aggregate the results of the experiments for k=2",
        default=False,
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="If set, the script will aggregate the results of the experiments for hash",
        default=False,
    )
    parser.add_argument(
        "--hashp",
        action="store_true",
        help="If set, the script will aggregate the results of the experiments for hash permutation",
        default=False,
    )
    
    return parser.parse_args()


datasets_names_old = [
    "rte",
    "cola",
    "sst2",
    "mrpc",
    "squad_v2",
    "rte-true",
    "mnli",
    "mnli-bis",
    "qnli",
    "qnli-bis",
    "qqp",
    "qqp-bis",
    "qqp-bis-bis"
]

datasets_names = [
    "squad_v2"
]

metric_dict = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "f1",
    "qqp": "f1",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
    "squad": "f1",
    "squad_v2": "f1",
    "mnli-bis": "accuracy",
    "qqp-bis": "f1",
    "qnli-bis": "accuracy",
    "rte-true": "accuracy",
    "qqp-bis-bis": "f1",
}


def get_best_result_json(all_results, train_log, dataset_name, spars_, k2):
    best_score = 0
    best_epoch = 0
    for ind, dict_ in enumerate(train_log["log_history"]):
        try:
            #score = dict_["eval_" + metric_dict[dataset_name]]
            score = dict_["best_" + metric_dict[dataset_name]]
            print(score)
            sparsity = dict_["eval_sparsity"]
        except KeyError:
            continue
        if not spars_ and sparsity != -1.0:
            if score > best_score:
                best_score = score
                best_epoch = dict_["epoch"]
        elif spars_ and k2 and sparsity != -1.0 and np.abs(sparsity - 0.5) < 0.1:
            if score > best_score:
                best_score = score
                best_epoch = dict_["epoch"]
        else:
            if score > best_score and np.abs(sparsity - 0.75) < 0.1 and sparsity != -1.0:
                best_score = score
                best_epoch = dict_["epoch"]

    if best_score == 0:
        best_score = all_results["best_" + metric_dict[dataset_name]]
        best_epoch = all_results["epoch"]
    return best_score, best_epoch


def generate_best_results_table(args):
    if args.advanced or args.perm or args.hash or args.hashp:
        num_experiments = 110
        range_of_exp = list(range(num_experiments + 1)) + [1011, 1012, 1013, 1014, 1015, 1021, 1022, 1023, 1024, 1025, 1031, 1032, 1033, 1034, 1035, 1041, 1042, 1043, 1044, 1045, 1051, 1052, 1053, 1054, 1055]
    elif args.ktwo:
        num_experiments = 100
        range_of_exp = [1011, 1012, 1013, 1014, 1015, 1021, 1022, 1023, 1024, 1025, 1031, 1032, 1033, 1034, 1035, 1041, 1042, 1043, 1044, 1045, 1051, 1052, 1053, 1054, 1055]
    else:
        num_experiments = 52
        range_of_exp = range(num_experiments + 1)

    
    dict_res = {
        "experiment": [i for i in range_of_exp],
    }

    for dataset_name in datasets_names:
        # check if the directory exists
        if dataset_name == "squad":
            continue
        else:
            dict_res_dataset = {
                "experiment": [i for i in range_of_exp],
                dataset_name + "_best_epoch": [],
                dataset_name + "_best_metric": [],
            }
            for experiment in range_of_exp:
                if args.perm:
                    path = f"results/{dataset_name}/moebert_perm_experiment_{experiment}/model"
                elif args.advanced:
                    path = f"results/{dataset_name}/moebert_experiment_{experiment}/model"
                    print(path)
                elif args.ktwo:
                    path = f"results/{dataset_name}/moebert_k2_experiment_{experiment}/model"
                elif args.hash:
                    path = f"results/{dataset_name}/moebert_hash_experiment_{experiment}/model"
                elif args.hashp:
                    path = f"results/{dataset_name}/moebert_hash_perm_experiment_{experiment}/model"
                else:
                    path = f"results/squad_experiment_{experiment}/model"
                try:
                    with open(path + "/all_results.json") as f:
                        all_results = json.load(f)
                except FileNotFoundError:
                    all_results = {
                        "best_" + metric_dict[dataset_name]: 0,
                        "epoch": 0,
                    }
                try:
                    if args.perm or args.hashp or args.hash:
                        with open(path + "/trainer_state_modified.json") as f:
                            train_log = json.load(f)
                    else:
                        with open(path + "/trainer_state.json") as f:
                            train_log = json.load(f)
                except FileNotFoundError:
                    train_log = {"log_history": []}
                best_score, best_epoch = get_best_result_json(
                    all_results, train_log, dataset_name, spars_=args.sparsity, k2=args.ktwo
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
            elif args.perm:
                if args.sparsity:
                    dataset_df.to_csv(
                        f"results/{dataset_name}/moebert_perm_results_aggregated_{dataset_name}_sparsity.csv",
                        index=False,
                    )
                else:
                    dataset_df.to_csv(
                        f"results/{dataset_name}/moebert_perm_results_aggregated_{dataset_name}.csv", index=False
                    )
            elif args.ktwo:
                if args.sparsity:
                    dataset_df.to_csv(
                        f"results/{dataset_name}/moebert_k2_results_aggregated_{dataset_name}_sparsity.csv",
                        index=False,
                    )
                else:
                    dataset_df.to_csv(
                        f"results/{dataset_name}/moebert_k2_results_aggregated_{dataset_name}.csv", index=False
                    )
            elif args.hash:
                if args.sparsity:
                    dataset_df.to_csv(
                        f"results/{dataset_name}/moebert_hash_results_aggregated_{dataset_name}_sparsity.csv",
                        index=False,
                    )
                else:
                    dataset_df.to_csv(
                        f"results/{dataset_name}/moebert_hash_results_aggregated_{dataset_name}.csv", index=False
                    )
            elif args.hashp:
                if args.sparsity:
                    dataset_df.to_csv(
                        f"results/{dataset_name}/moebert_hash_perm_results_aggregated_{dataset_name}_sparsity.csv",
                        index=False,
                    )
                else:
                    dataset_df.to_csv(
                        f"results/{dataset_name}/moebert_hash_perm_results_aggregated_{dataset_name}.csv",
                        index=False,
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
            elif args.perm:
                if args.sparsity:
                    df_fin = df_fin.merge(
                        pd.read_csv(
                            f"results/{dataset_name}/moebert_perm_results_aggregated_{dataset_name}_sparsity.csv"
                        ),
                        on="experiment",
                    )
                else:
                    df_fin = df_fin.merge(
                        pd.read_csv(f"results/{dataset_name}/moebert_perm_results_aggregated_{dataset_name}.csv"),
                        on="experiment",
                    )
            elif args.ktwo:
                if args.sparsity:
                    df_fin = df_fin.merge(
                        pd.read_csv(
                            f"results/{dataset_name}/moebert_k2_results_aggregated_{dataset_name}_sparsity.csv"
                        ),
                        on="experiment",
                    )
                else:
                    df_fin = df_fin.merge(
                        pd.read_csv(f"results/{dataset_name}/moebert_k2_results_aggregated_{dataset_name}.csv"),
                        on="experiment",
                    )
            elif args.hash:
                if args.sparsity:
                    df_fin = df_fin.merge(
                        pd.read_csv(
                            f"results/{dataset_name}/moebert_hash_results_aggregated_{dataset_name}_sparsity.csv"
                        ),
                        on="experiment",
                    )
                else:
                    df_fin = df_fin.merge(
                        pd.read_csv(f"results/{dataset_name}/moebert_hash_results_aggregated_{dataset_name}.csv"),
                        on="experiment",
                    )
            elif args.hashp:
                if args.sparsity:
                    df_fin = df_fin.merge(
                        pd.read_csv(
                            f"results/{dataset_name}/moebert_hash_perm_results_aggregated_{dataset_name}_sparsity.csv"
                        ),
                        on="experiment",
                    )
                else:
                    df_fin = df_fin.merge(
                        pd.read_csv(
                            f"results/{dataset_name}/moebert_hash_perm_results_aggregated_{dataset_name}.csv"
                        ),
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
    elif args.perm:
        if args.sparsity:
            df_fin.to_csv("results/moebert_perm_results_aggregated_sparsity.csv", index=False)
        else:
            df_fin.to_csv("results/moebert_perm_results_aggregated.csv", index=False)
    elif args.ktwo:
        if args.sparsity:
            df_fin.to_csv("results/moebert_k2_results_aggregated_sparsity.csv", index=False)
        else:
            df_fin.to_csv("results/moebert_k2_results_aggregated.csv", index=False)
    elif args.hash:
        if args.sparsity:
            df_fin.to_csv("results/moebert_hash_results_aggregated_sparsity.csv", index=False)
        else:
            df_fin.to_csv("results/moebert_hash_results_aggregated.csv", index=False)
    elif args.hashp:
        if args.sparsity:
            df_fin.to_csv("results/moebert_hash_perm_results_aggregated_sparsity.csv", index=False)
        else:
            df_fin.to_csv("results/moebert_hash_perm_results_aggregated.csv", index=False)
    else:
        df_fin.to_csv("results/results_aggregated.csv", index=False)


def generate_best_results_summary(args):
    if args.advanced:
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
    elif args.perm:
        if args.sparsity:
            moebert_sparsity = pd.read_csv("results/moebert_perm_results_aggregated_sparsity.csv")
            res_sparsity = pd.DataFrame(columns=["dataset", "best_epoch", "best_metric", "best_experiment"])
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
            res_sparsity.to_csv("results/moebert_perm_results_aggregated_sparsity_summary.csv", index=False)
        else:
            moebert = pd.read_csv("results/moebert_perm_results_aggregated.csv")
            res = pd.DataFrame(columns=["dataset", "best_epoch", "best_metric", "best_experiment"])
            for dataset_name in datasets_names:
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
            res.to_csv("results/moebert_perm_results_aggregated_summary.csv", index=False)
    elif args.ktwo:
        if args.sparsity:
            moebert_sparsity = pd.read_csv("results/moebert_k2_results_aggregated_sparsity.csv")
            res_sparsity = pd.DataFrame(columns=["dataset", "best_epoch", "best_metric", "best_experiment"])
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
            res_sparsity.to_csv("results/moebert_k2_results_aggregated_sparsity_summary.csv", index=False)
        else:
            moebert = pd.read_csv("results/moebert_k2_results_aggregated.csv")
            res = pd.DataFrame(columns=["dataset", "best_epoch", "best_metric", "best_experiment"])
            for dataset_name in datasets_names:
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
            res.to_csv("results/moebert_k2_results_aggregated_summary.csv", index=False)
    elif args.hash:
        if args.sparsity:
            moebert_sparsity = pd.read_csv("results/moebert_hash_results_aggregated_sparsity.csv")
            res_sparsity = pd.DataFrame(columns=["dataset", "best_epoch", "best_metric", "best_experiment"])
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
            res_sparsity.to_csv("results/moebert_hash_results_aggregated_sparsity_summary.csv", index=False)
        else:
            moebert = pd.read_csv("results/moebert_hash_results_aggregated.csv")
            res = pd.DataFrame(columns=["dataset", "best_epoch", "best_metric", "best_experiment"])
            for dataset_name in datasets_names:
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
            res.to_csv("results/moebert_hash_results_aggregated_summary.csv", index=False)
    elif args.hashp:
        if args.sparsity:
            moebert_sparsity = pd.read_csv("results/moebert_hash_perm_results_aggregated_sparsity.csv")
            res_sparsity = pd.DataFrame(columns=["dataset", "best_epoch", "best_metric", "best_experiment"])
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
            res_sparsity.to_csv("results/moebert_hash_perm_results_aggregated_sparsity_summary.csv", index=False)
        else:
            moebert = pd.read_csv("results/moebert_hash_perm_results_aggregated.csv")
            res = pd.DataFrame(columns=["dataset", "best_epoch", "best_metric", "best_experiment"])
            for dataset_name in datasets_names:
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
            res.to_csv("results/moebert_hash_perm_results_aggregated_summary.csv", index=False)
    else:
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



def main(args):
    os.makedirs("results/squad", exist_ok=True)

    generate_best_results_table(args)
    #if args.aggregate:
    generate_best_results_summary(args)


if __name__ == "__main__":
    args = parse_args()

    main(args)
