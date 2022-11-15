import argparse

import optuna
import torch

from data_utils import get_dataloaders, get_dataset
from nets import create_model
from training import train_epoch, EarlyStopping
from utils import print_args


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
        hyperparams = {
            "batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 1e-6,

            "channels_unet": trial.suggest_int("channels_unet", 256, 768, step=64),
            "depth": 3,
            "dropout_unet": 0.3,
            "activation_unet": "ELU",
            "pool_ratios": 0.8,
            "pooling": "TopKPooling",

            "readout": "cat",
            "layers_classifier": 2,
            "channels_classifier": 64,
            "dropout_classifier": 0.3,
            "activation_classifier": "ELU"
        }

        train_loader, val_loader = get_dataloaders(dataset, hyperparams["batch_size"])

        # Generate the model.
        model = create_model(
            in_channels=dataset.num_node_features,
            out_channels=dataset.num_classes,
            **hyperparams
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"],
        )
        criterion = torch.nn.CrossEntropyLoss()

        loss_history, acc_history = [], []
        with EarlyStopping(
                patience=patience, min_epochs=min_epochs, objective="maximize", verbose=True
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
    parser = argparse.ArgumentParser(description="Hyperparameter optimization")
    parser.add_argument("--device", type=str, default="cpu", help="device")
    parser.add_argument("--dataset", type=str, default="PROTEINS", help="dataset")
    parser.add_argument("--max-epochs", type=int, default=300, help="max epochs")
    parser.add_argument("--min-epochs", type=int, default=50, help="min epochs")
    parser.add_argument("--patience", type=int, default=50, help="early stopping")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--study-name", type=str, default="study")
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
    dataset = get_dataset(dataset)

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
