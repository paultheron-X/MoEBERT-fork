# This script will read the results jsons from the directory results for each dataset and each experiment and will create a csv file with the results

import json
import os
import pandas as pd
import argparse
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ds',
        help='Name of the dataset',
    )
    parser.add_argument(
        '--number',
        '-n',
        help='Number of the experiment')
    parser.add_argument(
        '--exp',
        '-e',
        help='Name of the experiment',
        default="moebert_experiment",
        choices=["moebert_experiment", "moebert_hash_experiment", "moebert_hash_perm_experiment", "moebert_perm_experiment", "moebert_k2_experiment"]
    )
    return parser.parse_args()


datasets_names = ["cola", "sst-2", "mrpc", "qqp", "mnli", "qnli", "rte", "squad", "mnli-bis", "qqp-bis", "qnli-bis", "rte-true"]

def generate_curve_i(dataset_name, i, exp):
    path = f"results/{dataset_name}/{exp}_{i}/model/trainer_state_modified.json"
    try:
        with open(path, "r") as f:
            train_log = json.load(f)
        epochs = [-1]
        losses = [-1]
        lr = [-1]
        for ind, dict_ in enumerate(train_log["log_history"]):
            try:
                if dict_["epoch"] >= epochs[-1]:
                    lr.append(dict_["learning_rate"])
                    s = dict_["sparsity"]
                    epochs.append(dict_["epoch"])
                    losses.append(dict_["loss"])
                else:
                    epochs = []
                    losses = []
                    lr = []
                    lr.append(dict_["learning_rate"])
                    s = dict_["sparsity"]
                    epochs.append(dict_["epoch"])
                    losses.append(dict_["loss"])
                
            except KeyError:
                continue
        sns.lineplot(x=epochs, y=losses, label=f"Experiment {i}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss curve for {dataset_name}")
        plt.savefig(f"results/{dataset_name}/{exp}_{i}/loss_curve.png")
        plt.clf()
        
    except FileNotFoundError:
        pass

def generate_curve(dataset_name, exp):
    for i in range(1, 100):
        generate_curve_i(dataset_name, i, exp)
        
    
if __name__ == "__main__":
    args  = parse_args()
    logging.basicConfig(level=logging.INFO)
    
    dataset_name = args.ds
    if dataset_name == "all":
        for dataset_name in datasets_names:
            logging.info(f"Generating loss curves for dataset: {dataset_name}, all experiments")
            generate_curve(dataset_name, exp=args.exp)
    else:
        if args.number:
            logging.info(f"Generating loss curves for dataset: {dataset_name}, experiment {args.number}")
            generate_curve_i(dataset_name, int(args.number), exp=args.exp)
        else:
            logging.info(f"Generating loss curves for dataset: {dataset_name}, all experiments")
            generate_curve(dataset_name, exp=args.exp)
        