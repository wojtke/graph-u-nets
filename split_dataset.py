import argparse
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.datasets import TUDataset

from utils import NumpyEncoder, set_reproducibility


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
    if args.dataset in ["PROTEINS", "ENZYMES"]:
        dataset = TUDataset(root='data', name=args.dataset)
    else:
        raise ValueError(f"Dataset {args.dataset} not found.")

    # split data into stratified cv folds
    if args.method == "cv":
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True)
        folds = skf.split(np.arange(dataset.len()), [data.y.item() for data in dataset])

        for train_idx, test_idx in folds:
            with open(f'data/{args.dataset}/split.json', 'w') as f:
                json.dump({
                    "train_idx": train_idx,
                    "test_idx": test_idx
                }, f, cls=NumpyEncoder)

    # split data into train and test set using stratified holdout
    elif args.method == "holdout":
        train_idx, test_idx = train_test_split(
            np.arange(dataset.len()),
            test_size=args.test_size,
            stratify=[data.y.item() for data in dataset],
        )

        with open(f'data/{args.dataset}/split.json', 'w') as f:
            json.dump({
                "train_idx": train_idx,
                "test_idx": test_idx
            }, f, cls=NumpyEncoder)
