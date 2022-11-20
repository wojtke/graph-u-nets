import argparse

import optuna
import torch
from torch.utils.tensorboard import SummaryWriter

from data_utils import get_dataloaders, get_dataset
from nets import create_model
from training import Trainer
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
            "batch_size": trial.suggest_int("batch_size", 32, 128, step=32),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 5e-3, log=True),

            "channels_unet": trial.suggest_int("channels_unet", 256, 768, step=64),
            "depth": trial.suggest_int("depth", 3, 4),
            "dropout_unet": trial.suggest_float("dropout_unet", 0.0, 0.3),
            "activation_unet": "ELU",
            "pool_ratios": trial.suggest_float("pool_ratios", 0.5, 0.9),
            "pooling": "TopKPooling",

            "readout": trial.suggest_categorical("readout", ["add", "mean", "max", "cat"]),
            "layers_classifier": 2,
            "channels_classifier": trial.suggest_int("channels_classifier", 64, 128, step=32),
            "dropout_classifier": trial.suggest_float("dropout_classifier", 0.1, 0.3),
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

        writer = SummaryWriter(f'runs/{trial.study.study_name}/trial{trial.number:03d}')

        trainer = Trainer(model, optimizer, criterion, device, writer=writer)
        trainer.set_early_stopping(patience=patience, min_epochs=min_epochs, objective="acc")
        trainer.set_optuna_trial_prunning(trial, objective="acc")
        trainer.train(train_loader, val_loader, epochs=max_epochs)

        return trainer.get_best("val_acc")

    return objective


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization")
    parser.add_argument("--study-name", type=str, default="study5")
    parser.add_argument("--dataset", type=str, default="PROTEINS", help="dataset")
    parser.add_argument("--storage", type=str, default="sqlite:///storage.db")
    parser.add_argument("--max-epochs", type=int, default=400, help="max epochs")
    parser.add_argument("--min-epochs", type=int, default=50, help="min epochs")
    parser.add_argument("--patience", type=int, default=50, help="early stopping")
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu", help="device")
    parser.add_argument("--n-jobs", type=int, default=1)
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
