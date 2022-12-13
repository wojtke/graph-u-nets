import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from data_utils import load_dataset, load_splits, split_indices, load_dataset_artifacts
from hyperparams import HyperparamsSpace
from model_selection import select_hyperparams
from models import GNN
from training import Trainer
from utils import set_reproducibility, print_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="PROTEINS", help="dataset")
    parser.add_argument("--pool", type=str, default="topk", help="Pooling method")
    parser.add_argument("--conv", type=str, default="gcn", help="Convolution method")
    parser.add_argument("--selection-trials", type=int, default=2, help="Number of trials for hyperparameter selection")
    parser.add_argument("--test-trials", type=int, default=2, help="Number of trials for model testing")
    parser.add_argument("--hyperparams-space", type=str, default="hyperparams_space.yml")
    parser.add_argument("--n-jobs", type=int, default=2, help="Number of parallel jobs for hyperparameter selection")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device to be used for training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    study_name = f"{args.dataset}/{args.conv}_{args.pool}"

    set_reproducibility(args.seed)
    device = torch.device(int(args.device)) if args.device.isdigit() else torch.device(args.device)

    # Load the dataset related things
    dataset = load_dataset(args.dataset)
    task, evaluation_metric = load_dataset_artifacts(args.dataset)
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
    for i, (train_idx, test_idx) in enumerate(splits):
        # Split the train dataset into train and val.
        train_idx, val_idx = split_indices(
            dataset, test_size=0.1, indices=train_idx, method="stratified" if task == "classification" else "random"
        )

        # Select the best hyperparameters using optuna.

        hyperparams = select_hyperparams(
            dataset,
            split=(train_idx, val_idx),
            study_name=f"{study_name}/{i}",
            hyperparams_space=hyperparams_space,
            metric=evaluation_metric,
            n_trials=args.selection_trials,
            task=task,
            pruning=False,
            n_jobs=args.n_jobs,
            device=device,
        )
        print(f"Selected hyperparams: \n{hyperparams}")

        inner_results = []
        for j in range(args.test_trials):
            # re-split the dataset and create the dataloaders
            train_idx, val_idx = split_indices(
                dataset,
                test_size=0.1,
                indices=train_idx + val_idx,
                method="stratified" if task == "classification" else "random",
            )
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
            writer = SummaryWriter(f"runs/{study_name}/{i}/assessment{j}")
            trainer = Trainer(model, optimizer, criterion, metric=evaluation_metric, device=device, writer=writer)
            trainer.set_early_stopping(patience=hyperparams.patience, min_epochs=hyperparams.min_epochs)
            trainer.train(train_loader, val_loader, epochs=hyperparams.max_epochs)

            # Evaluate the model on the test set.
            test_loader = DataLoader(dataset[test_idx], hyperparams.batch_size, shuffle=False)
            evaluation_score = trainer.evaluate(test_loader)

            inner_results.append(evaluation_score)
            print(f"Test trial {j + 1}/{args.test_trials} - {evaluation_metric.__name__}: {evaluation_score:.4f}")
        results.append(sum(inner_results) / len(inner_results))

    print("========================================")
    print("Results: ", results)
    print(f"Average score ({evaluation_metric.__name__}): {sum(results) / len(results)}")
    print(f"Standard deviation: {np.std(results)}")

    with open(f"runs/{study_name}/result", "w") as f:
        f.write(
            f"Results: {results}\n"
            f"Average score ({evaluation_metric.__name__}): {sum(results) / len(results)}\n"
            f"Standard deviation: {np.std(results)}\n"
        )
