import pandas as pd
import argparse


hyper_params = pd.read_csv("sh_scripts/python_helpers/moebert_distil_grid.csv", sep=";", index_col=0, dtype={'experiment_name': int, 'batch_size': int, 'learning_rate': float, 'weight_decay': float, 'entropy': float, 'gamma': float, 'distill': float})


def _parse_args():
    parser = argparse.ArgumentParser(description="Launch a job from the grid")
    parser.add_argument(
        "--num",
        "-n",
        default="1",
        help="Number of the job to launch",
    )
    return parser.parse_args()


args = _parse_args()

jobs_params = hyper_params.loc[int(args.num)]

print(
    f"{jobs_params.name} {jobs_params.batch_size} {jobs_params.learning_rate} {jobs_params.weight_decay} {jobs_params.entropy} {jobs_params.gamma} {jobs_params.distill}"
)
