import argparse

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from data_utils import load_dataset, load_splits, split_dataset
from hyperparams import HyperparamsSpace
from model_selection import select_hyperparams
from models import GNN
from training import Trainer
from utils import set_reproducibility


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="PROTEINS", help="dataset")
    parser.add_argument("--pool", type=str, default="topk", help="Pooling method")
    parser.add_argument("--conv", type=str, default="gcn", help="Convolution method")
    parser.add_argument(
        "--selection-trials", type=int, default=100, help="Number of trials for hyperparameter selection"
    )
    parser.add_argument("--test-trials", type=int, default=3, help="Number of trials for model testing")
    parser.add_argument("--hyperparams-space", type=str, default="hyperparams_space.yml")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    set_reproducibility(args.seed)

    # Load the dataset related things
    dataset, task, evaluation_metric = load_dataset(args.dataset)
    splits = load_splits(args.dataset)
    print(f"Dataset loaded: {args.dataset} - Task: {task} - Metric: {evaluation_metric}")
    print(f"Loaded {len(splits)} splits. Train: {len(splits[0][0])} examples - Val: {len(splits[0][1])} examples")

    hyperparams_space = HyperparamsSpace.load(
        path=args.hyperparams_space,
        conv=args.conv,
        pool=args.pool,
    )
    print(f"Hyperparams space: \n{hyperparams_space}")

    results = []
    # For each split run model selection and testing.
    for train_idx, test_idx in splits:
        # Split the train dataset into train and val.
        train_idx, val_idx = split_dataset(dataset, test_size=0.1, indices=train_idx, return_indices=True)

        # Select the best hyperparameters using optuna.
        hyperparams = select_hyperparams(
            dataset,
            split=(train_idx, val_idx),
            study_name=f"{args.dataset}/{args.conv}_{args.pool}",
            hyperparams_space=hyperparams_space,
            metric=evaluation_metric,
            n_trials=1,
            task=task,
            pruning=False,
        )
        print(f"Selected hyperparams: \n{hyperparams}")

        inner_results = []
        for i in range(args.test_trials):
            # re-split the dataset and create the dataloaders
            train_idx, val_idx = split_dataset(dataset, test_size=0.1, indices=train_idx + val_idx, return_indices=True)
            train_loader = DataLoader(dataset[train_idx], hyperparams.batch_size, shuffle=True)
            val_loader = DataLoader(dataset[val_idx], hyperparams.batch_size, shuffle=False)

            # Initialize the model and the trainer using selected hyperparams
            model = GNN(in_channels=dataset.num_features, out_channels=dataset.num_classes, hyperparams=hyperparams)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=hyperparams.lr,
                weight_decay=hyperparams.weight_decay,
            )
            criterion = torch.nn.MSELoss() if task == "regression" else torch.nn.CrossEntropyLoss()

            # Train the model. Use early stopping on the validation set.
            trainer = Trainer(model, optimizer, criterion, metric=evaluation_metric)
            trainer.set_early_stopping(patience=hyperparams.patience, min_epochs=hyperparams.min_epochs)
            trainer.train(train_loader, val_loader, epochs=hyperparams.max_epochs)

            # Evaluate the model on the test set.
            test_loader = DataLoader(dataset[test_idx], hyperparams.batch_size, shuffle=False)
            evaluation_score = trainer.evaluate(test_loader)

            inner_results.append(evaluation_score)
            print(f"Test trial {i + 1}/{args.test_trials} - {evaluation_metric.__name__}: {evaluation_score:.4f}")
        results.append(sum(inner_results) / len(inner_results))

    print("========================================")
    print("Results: ", results)
    print(f"Average score ({evaluation_metric.__name__}): {sum(results) / len(results)}")
    print(f"Standard deviation: {np.std(results)}")
