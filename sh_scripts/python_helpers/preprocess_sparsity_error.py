import json
import os
import pandas as pd
import argparse
import numpy as np


datasets_names = [
    "rte",
    "cola",
    "sst2",
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
    "sst2": "accuracy",
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

# load the trainer first stage logs for the first stage finetuning

def read_base_log(dataset):
    base_log_path = f"results/{dataset}/experiment_{dataset}_finetuned_model/model/trainer_state.json"
    try:
        with open(base_log_path, "r") as f:
            train_log = json.load(f)
    except FileNotFoundError:
            train_log = {"log_history": []}
    return train_log
            
def read_experiment_log(dataset, experiment_name):
    experiment_log_path = f"results/{dataset}/moebert_perm_experiment_{experiment_name}/model/trainer_state.json"
    try:
        with open(experiment_log_path, "r") as f:
            train_log = json.load(f)
    except FileNotFoundError:
            train_log = {"log_history": []}
    return train_log

for dataset in datasets_names:
    for experiment_name in range(1, 100):
        try:
            base_log = read_base_log(dataset)
            experiment_log = read_experiment_log(dataset, experiment_name)
            for i,a in enumerate(base_log["log_history"]):
                for j,b in enumerate(experiment_log["log_history"]):
                    if a["epoch"] == b["epoch"]:
                        try:
                            if a["eval_loss"] == b["eval_loss"]:
                                b['eval_sparsity'] = -1
                            else:
                                '''if dataset in ["rte", "sst2", "qnli"] and experiment_name<30:
                                    b['eval_sparsity'] = 0.75
                                elif dataset in ["mrpc"] and experiment_name<22:
                                    b['eval_sparsity'] = 0.75
                                else:'''
                                pass                                
                        except KeyError:
                            pass
            with open(f"results/{dataset}/moebert_perm_experiment_{experiment_name}/model/trainer_state_modified.json", "w") as f:
                json.dump(experiment_log, f)
        except FileNotFoundError:
            pass
                    