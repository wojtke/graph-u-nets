import argparse

import optuna
import torch
from torch_geometric.datasets import TUDataset

from data_utils import get_dataloaders
from nets import GNN, GraphUNet
from training import train_epoch, EarlyStopping
from utils import print_args


def define_model(trial, in_channels, out_channels):
    """Define the model to be optimized.

    Args:
        trial (optuna.trial.Trial): Current trial.
        in_channels (int): Number of input channels - number of features per node.
        out_channels (int): Number of output channels - number of classes.
    """
    g_unet = GraphUNet(
        in_channels=in_channels,
        hidden_channels=trial.suggest_int("hidden_channels", 256, 768, step=64),
        out_channels=trial.suggest_int("out_channels", 64, 256, step=64),
        depth=3,  # trial.suggest_int('depth', 2, 4),
        pool_ratios=trial.suggest_float("pool_ratios", 0.5, 0.9),
        sum_res=False,  # trial.suggest_categorical('sum_res', [True, False]),
        act="ReLU",  # trial.suggest_categorical('act', ["ReLU", "LeakyReLU"]),
        pool="TopKPooling",
    )

    readout = trial.suggest_categorical("readout", ["add", "mean", "max"])
    mlp_layers = trial.suggest_int("mlp_layers", 1, 3)
    mlp_channels = trial.suggest_int("mlp_channels", 32, 128, step=16)
    mlp = torch.nn.Sequential()
    mlp.add_module("mlp_0", torch.nn.Linear(g_unet.out_channels, mlp_channels))
    mlp.add_module("mlp_act_0", torch.nn.ReLU())
    for i in range(1, mlp_layers):
        mlp.add_module(f"mlp_{i}", torch.nn.Linear(mlp_channels, mlp_channels))
        mlp.add_module(f"mlp_act_{i}", torch.nn.ReLU())
    mlp.add_module(f"mlp_{mlp_layers}", torch.nn.Linear(mlp_channels, out_channels))

    return GNN(g_unet, readout, mlp)


def define_objective(dataset, device, max_epochs, min_epochs, patience):
    """Define the objective function to be optimized.

    Args:
        dataset (torch_geometric.data.Dataset): Dataset.
        device (str): Device.
        max_epochs (int): Maximum number of epochs.
        min_epochs (int): Minimum number of epochs.
        patience (int): Patience for early stopping.
    """

    def objective(trial):
        torch.cuda.empty_cache()
        batch_size = trial.suggest_int("batch_size", 32, 128, step=32)

        train_loader, val_loader = get_dataloaders(dataset, batch_size)

        # Generate the model.
        model = define_model(trial, dataset.num_features, dataset.num_classes)
        model = model.to(device)

        # Generate the optimizers.
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        )

        criterion = torch.nn.CrossEntropyLoss()

        loss_history, acc_history = [], []
        # Training of the model.
        with EarlyStopping(
            patience=patience, min_epochs=min_epochs, objective="maximize"
        ) as es:
            for epoch in range(max_epochs):
                loss, acc = train_epoch(
                    model, optimizer, criterion, train_loader, val_loader, device
                )
                loss_history.append(loss)
                acc_history.append(acc)
                es.check(acc)

                trial.report(acc, epoch)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return max(acc_history)

    return objective


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna example: pytorch")
    parser.add_argument("--device", type=str, default="cpu", help="device")
    parser.add_argument("--dataset", type=str, default="PROTEINS", help="dataset")
    parser.add_argument("--max-epochs", type=int, default=200, help="max epochs")
    parser.add_argument("--min-epochs", type=int, default=50, help="min epochs")
    parser.add_argument(
        "--patience", type=int, default=50, help="patience for early stopping"
    )
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--study-name", type=str, default="graph-unets")
    parser.add_argument("--storage", type=str, default="sqlite:///storage.db")
    args = parser.parse_args()
    return args


def main(
    device,
    dataset,
    n_trials,
    timeout,
    n_jobs,
    study_name,
    storage,
    max_epochs,
    min_epochs,
    patience,
):
    dataset = TUDataset(root="data/TUDataset", name=dataset)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=0),
        load_if_exists=True,
    )

    study.optimize(
        define_objective(dataset, device, max_epochs, min_epochs, patience),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
    )


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(**vars(args))
