import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

from data_utils import load_dataset, load_dataset_artifacts, split_indices
from utils import set_reproducibility, mkdir_if_not_exists


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="PROTEINS", help="Dataset name")
    parser.add_argument("--name", type=str, default="default", help="Split name (default: 'default')")
    parser.add_argument("--method", type=str, default="cv", choices=["cv", "holdout"], help="Split method")
    parser.add_argument("--folds", type=int, default=10, help="Number of folds for cross-validation")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test size for holdout")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    set_reproducibility(args.seed)

    # load the dataset
    dataset = load_dataset(args.dataset)

    mkdir_if_not_exists(f"data/{args.dataset}/splits/{args.name}")

    # split data into stratified cv folds
    if args.method == "cv":
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True)
        folds = skf.split(np.arange(dataset.len()), [data.y.item() for data in dataset])

        for i, (train_idx, test_idx) in enumerate(folds):
            pd.DataFrame(train_idx).to_csv(f"data/{args.dataset}/splits/{args.name}/train{i}.csv", index=False)
            pd.DataFrame(test_idx).to_csv(f"data/{args.dataset}/splits/{args.name}/test{i}.csv", index=False)
            print(f"Fold {i} saved to data/{args.dataset}/splits/{args.name}")

    # split data into train and test set using stratified holdout
    elif args.method == "holdout":
        train_idx, test_idx = split_indices(
            dataset, test_size=args.test_size, method="stratified"
        )

        pd.DataFrame(train_idx).to_csv(f"data/{args.dataset}/splits/{args.name}/train.csv", index=False)
        pd.DataFrame(test_idx).to_csv(f"data/{args.dataset}/splits/{args.name}/test.csv", index=False)
        print(f"Train and test splits saved to data/{args.dataset}/splits/{args.name}")
