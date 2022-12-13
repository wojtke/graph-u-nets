import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

from data_utils import load_dataset, load_dataset_artifacts, split_indices
from utils import set_reproducibility, mkdir_if_not_exists


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="PROTEINS")
    parser.add_argument("--method", "--m", type=str, default="cv", choices=["cv", "holdout"])
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    set_reproducibility(args.seed)

    # load the dataset
    dataset = load_dataset(args.dataset)
    task, evaluation_metric = load_dataset_artifacts(args.dataset)

    mkdir_if_not_exists(f"data/{args.dataset}/splits")

    # split data into stratified cv folds
    if args.method == "cv":
        if task == "regression":
            kf = KFold(n_splits=args.folds, shuffle=True)
            folds = kf.split(np.arange(dataset.len()))
        else:
            skf = StratifiedKFold(n_splits=args.folds, shuffle=True)
            folds = skf.split(np.arange(dataset.len()), [data.y.item() for data in dataset])

        for i, (train_idx, test_idx) in enumerate(folds):
            pd.DataFrame(train_idx).to_csv(f"data/{args.dataset}/splits/train{i}.csv", index=False)
            pd.DataFrame(test_idx).to_csv(f"data/{args.dataset}/splits/test{i}.csv", index=False)
            print(f"Fold {i} saved to data/{args.dataset}/splits/")

    # split data into train and test set using stratified holdout
    elif args.method == "holdout":
        train_idx, test_idx = split_indices(
            dataset, test_size=args.test_size, method="stratified" if task == "classification" else None
        )

        pd.DataFrame(train_idx).to_csv(f"data/{args.dataset}/splits/train.csv", index=False)
        pd.DataFrame(test_idx).to_csv(f"data/{args.dataset}/splits/test.csv", index=False)
        print(f"Train and test splits saved to data/{args.dataset}/splits/")
