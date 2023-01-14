import gc
from typing import Tuple, Callable

import optuna
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from callbacks import TensorBoardWriter, EarlyStopping, OptunaPruning
from dataset_utils import get_out_channels, get_criterion
from hyperparams import HyperparamsSpace, Hyperparams
from models import GNN
from training import Trainer
from utils import mkdir_if_not_exists


def define_objective(
        dataset: Dataset,
        split: Tuple[list, list],
        hyperparams_space: HyperparamsSpace,
        study_name: str,
        pruning: bool,
) -> Callable:
    """Define the objective function to be optimized.

    Args:
        study_name: Name of the study.
        dataset: Dataset to be used for training.
        split: Split of the dataset to be used for training and validation.
        hyperparams_space: Hyperparameters space.
        pruning: Whether to use pruning or not.
    """

    def objective(trial: optuna.Trial) -> float:
        """Objective function to be optimized by optuna."""
        gc.collect()
        torch.cuda.empty_cache()

        hyperparams = hyperparams_space.pick(trial)
        hyperparams.save(f"runs/{study_name}/hpo/{trial.number}")

        train_idx, val_idx = split
        train_loader = DataLoader(dataset[list(train_idx)], hyperparams.batch_size, shuffle=True)
        val_loader = DataLoader(dataset[list(val_idx)], hyperparams.batch_size, shuffle=False)

        model = GNN(
            in_channels=dataset.num_features,
            out_channels=get_out_channels(dataset),
            hyperparams=hyperparams
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.lr, weight_decay=hyperparams.weight_decay)
        criterion = get_criterion(dataset)

        trainer = Trainer(model, optimizer, criterion, dataset.eval_metric)
        trainer.add_callbacks(
            EarlyStopping(patience=hyperparams.patience),
            #TensorBoardWriter(log_dir=f"runs/{study_name}/hpo/{trial.number}"),
        )
        if pruning:
            trainer.add_callbacks(OptunaPruning(trial))

        trainer.train(train_loader, val_loader, epochs=hyperparams.max_epochs)

        return trainer.history.get_best()

    return objective


def select_hyperparams(
        dataset: Dataset,
        split: Tuple[list, list],
        study_name: str,
        hyperparams_space: HyperparamsSpace,
        pruning: bool = True,
        n_trials: int = 10,
) -> Hyperparams:
    """Select the best hyperparameters for the given dataset and split.

    Args:
        device:
        dataset: Dataset to be used for training.
        split: Split of the dataset to be used for training and validation.
        study_name: Name of the study.
        hyperparams_space: Hyperparameters space.
        pruning: Whether to use pruning or not.
        n_trials: Number of trials to be performed.
    """

    study = optuna.create_study(
        study_name=study_name,
        direction=dataset.eval_metric.direction(),
        sampler=optuna.samplers.TPESampler(seed=0),
        load_if_exists=True,
    )

    mkdir_if_not_exists(f"runs/{study_name}/hpo")
    hyperparams_space.save(f"runs/{study_name}/hyperparams_space.yml")

    objective = define_objective(
        dataset,
        split,
        study_name=study_name,
        hyperparams_space=hyperparams_space,
        pruning=pruning
    )

    study.optimize(objective, n_trials=n_trials)

    best_hyperparams = Hyperparams(
        **study.best_params,
        **{k: v for k, v in hyperparams_space.__dict__.items() if k not in study.best_params},
    )

    best_hyperparams.save(f"runs/{study_name}/best_hyperparams.yml")

    return best_hyperparams
