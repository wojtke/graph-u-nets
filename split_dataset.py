import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.data import Dataset

from dataset_utils import load_tud

from utils import set_reproducibility, mkdir_if_not_exists


def load_splits(dataset_name: str, split_name: str = "default"):
    """Loads saved train and test splits for a dataset.

    Note: Need to run split_dataset.py first to generate the splits.

    Args:
        dataset_name (str): Name of the dataset to load splits for.
        split_name (str): Name of the split to load (default: "default").
    """
    splits = []
    for file_name in os.listdir(f"data/{dataset_name}/splits/{split_name}"):
        if file_name.startswith("train"):
            with open(f"data/{dataset_name}/splits/{split_name}/{file_name}", "r") as f:
                train_idx = pd.read_csv(f, header=None).values.flatten()
            with open(f"data/{dataset_name}/splits/{split_name}/{file_name.replace('train', 'test')}", "r") as f:
                test_idx = pd.read_csv(f, header=None).values.flatten()
            splits.append((list(train_idx), list(test_idx)))
    return splits


def split_indices(dataset: Dataset, test_size=0.1, indices=None, method=None):
    """Returns train and test indices for a dataset.

    Args:
        dataset (torch_geometric.data.Dataset): Dataset to split into train and test.
        test_size (float): Proportion of the dataset to use for testing (default: 0.2).
        indices (list): Indices to use for splitting (default: None).
        method: Method to use for splitting (default: None).
    """
    if indices is None:
        indices = np.arange(dataset.len())

    stratify = [data.y.item() for data in dataset[indices]] if method == "stratified" else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=stratify
    )

    return list(train_idx), list(test_idx)


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
    dataset = load_tud(args.dataset)

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
