import gc
from typing import Tuple, Callable

import optuna
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from hyperparams import HyperparamsSpace, Hyperparams
from metrics import Metric
from models import GNN
from training import Trainer
from utils import mkdir_if_not_exists


def define_objective(
    dataset: Dataset,
    split: Tuple[list, list],
    hyperparams_space: HyperparamsSpace,
    evaluation_metric: Metric,
    task: str,
    pruning: bool = True,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Callable:
    """Define the objective function to be optimized.

    Args:
        dataset: Dataset to be used for training.
        split: Split of the dataset to be used for training and validation.
        hyperparams_space: Hyperparameters space.
        evaluation_metric: Metric to be used for model evaluation.
        task: Task to be performed (classification or regression).
        pruning: Whether to use pruning or not.
        device: Device to use. (default: "cuda")
    """

    def objective(trial: optuna.Trial) -> float:
        """Objective function to be optimized by optuna."""

        gc.collect()
        torch.cuda.empty_cache()

        hyperparams = hyperparams_space.pick(trial)

        print(f"Hyperparameters:\n {hyperparams}")

        train_idx, val_idx = split
        train_loader = DataLoader(dataset[list(train_idx)], hyperparams.batch_size, shuffle=True)
        val_loader = DataLoader(dataset[list(val_idx)], hyperparams.batch_size, shuffle=False)

        # Generate the model.
        out_channels = 1 if task == "regression" else dataset.num_classes
        model = GNN(in_channels=dataset.num_features, out_channels=out_channels, hyperparams=hyperparams)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hyperparams.lr,
            weight_decay=hyperparams.weight_decay,
        )
        criterion = torch.nn.MSELoss() if task == "regression" else torch.nn.CrossEntropyLoss()

        writer = SummaryWriter(f"runs/{trial.study.study_name}/trial{trial.number:04d}")

        trainer = Trainer(model, optimizer, criterion, evaluation_metric, writer=writer, device=device)
        trainer.set_early_stopping(patience=hyperparams.patience, min_epochs=hyperparams.min_epochs)
        if pruning:
            trainer.set_optuna_trial_pruning(trial)
        trainer.train(train_loader, val_loader, epochs=hyperparams.max_epochs)

        return trainer.get_best_metric_score()

    return objective


def select_hyperparams(
    dataset: Dataset,
    split: Tuple[list, list],
    study_name: str,
    hyperparams_space: HyperparamsSpace,
    metric: Metric,
    task: str,
    pruning: bool = True,
    n_trials: int = 10,
    n_jobs: int = 1,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Hyperparams:
    """Select the best hyperparameters for the given dataset and split.

    Args:
        device:
        dataset: Dataset to be used for training.
        split: Split of the dataset to be used for training and validation.
        study_name: Name of the study.
        hyperparams_space: Hyperparameters space.
        metric: Metric to be used for model evaluation.
        task: Task to be performed (classification or regression).
        pruning: Whether to use pruning or not.
        n_trials: Number of trials to be performed.
        n_jobs: Number of parallel jobs. (default: 1)
    """

    study = optuna.create_study(
        study_name=f"{study_name}",
        direction=metric.direction(),
        sampler=optuna.samplers.TPESampler(seed=0),
        load_if_exists=True,
    )

    mkdir_if_not_exists(f"runs/{study_name}")
    hyperparams_space.save(f"runs/{study_name}/hyperparams_space.yml")

    objective = define_objective(
        dataset,
        split,
        hyperparams_space=hyperparams_space,
        evaluation_metric=metric,
        task=task,
        pruning=pruning,
        device=device,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    best_hyperparams = Hyperparams(
        **study.best_params,
        **{k: v for k, v in hyperparams_space.__dict__.items() if k not in study.best_params},
    )

    best_hyperparams.save(f"runs/{study_name}/best_hyperparams.yml")

    return best_hyperparams
