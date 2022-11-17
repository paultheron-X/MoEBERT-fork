# This script will read the results jsons from the directory results_gcloud for each dataset and each experiment and will create a csv file with the results

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
    return parser.parse_args()


datasets_names = ["cola", "sst-2", "mrpc", "qqp", "mnli", "qnli", "rte", "squad", "mnli-bis", "qqp-bis", "qnli-bis", "rte-true"]

def generate_curve_i(dataset_name, i):
    path = f"results_gcloud/{dataset_name}/moebert_experiment_{i}/trainer_state.json"
    try:
        with open(path, "r") as f:
            train_log = json.load(f)
        epochs = []
        losses = []
        lr = []
        for ind, dict_ in enumerate(train_log["log_history"]):
            try:
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
        plt.savefig(f"results_gcloud/{dataset_name}/moebert_experiment_{i}/loss_curve.png")
        plt.clf()
        
    except FileNotFoundError:
        pass

def generate_curve(dataset_name):
    for i in range(1, 100):
        generate_curve_i(dataset_name, i)
        
    
if __name__ == "__main__":
    args  = parse_args()
    logging.basicConfig(level=logging.INFO)
    
    dataset_name = args.ds
    if dataset_name == "all":
        for dataset_name in datasets_names:
            logging.info(f"Generating loss curves for dataset: {dataset_name}, all experiments")
            generate_curve(dataset_name)
    else:
        if args.number:
            logging.info(f"Generating loss curves for dataset: {dataset_name}, experiment {args.number}")
            generate_curve_i(dataset_name, int(args.number))
        else:
            logging.info(f"Generating loss curves for dataset: {dataset_name}, all experiments")
            generate_curve(dataset_name)
        