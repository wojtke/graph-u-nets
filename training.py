import numpy as np
import optuna
import torch
from torch_geometric.loader import DataLoader

from metrics import Accuracy, Metric


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: torch.nn,
            metric: Metric = Accuracy,
            writer=None,
            verbose: bool = False,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """Initialize trainer.

        Args:
            model: Model to train.
            optimizer: Optimizer to use.
            criterion: Loss function.
            metric: Metric to use for evaluation. (default: Accuracy)
            writer: Tensorboard writer. (default: None)
            verbose: Print training progress. (default: False)
            device: Device to use. (default: "cuda")
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.metric = metric

        self.epoch = 0
        self.history = {"train_loss": [], "val_loss": [], "train_metric": [], "val_metric": []}

        self.writer = writer
        self.verbose = verbose

        self.early_stopping = False
        self.patience, self.min_epochs = None, None
        self.best_epoch, self.best_value = None, None
        self.save_path = None
        self._early_stopping_check = None
        self._optuna_callback = None

    def set_early_stopping(self, patience: int, min_epochs: int = 0, save_path: str = None):
        """Set early stopping parameters.

        Args:
            patience: Number of epochs to wait if there is no improvement.
            min_epochs: Minimum number of epochs to train. (default: 0)
            save_path: Path to save best model weights. (default: None)
        """

        self.early_stopping = True
        self.patience = patience
        self.min_epochs = min_epochs
        self.save_path = save_path

        self.best_epoch = 1
        self.best_value = np.inf if self.metric.direction() == "minimize" else -np.inf

        def check(scores: dict):
            if (
                    self.metric.direction() == "minimize"
                    and scores["val_metric"] < self.best_value
                    or self.metric.direction() == "maximize"
                    and scores["val_metric"] > self.best_value
            ):
                self.best_epoch = self.epoch
                self.best_value = scores["val_metric"]
                if self.save_path:
                    torch.save(self.model.state_dict(), self.save_path)

        self._early_stopping_check = check

    def set_optuna_trial_pruning(self, trial: optuna.Trial):
        """Set optuna trial pruning callback."""

        def optuna_callback(scores):
            trial.report(scores["val_metric"], self.epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        self._optuna_callback = optuna_callback

    def get_best_metric_score(self) -> float:
        """Get best metric score on validation set."""
        return self.history["val_metric"][self.best_epoch - 1]

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        """Train model.

        Args:
            train_loader: Train data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs to train; if early stopping is enabled, this is the maximum number of epochs.
        """
        while self.epoch < epochs:
            self.epoch += 1
            scores = self.train_epoch(train_loader, val_loader)

            if self.verbose:
                print(
                    f"Epoch {self.epoch}. "
                    f"Loss - val/train: {scores['val_loss']:.4f}/{scores['train_loss']:.4f}. "
                    f"{self.metric.__name__} - val/train: {scores['val_metric']:.4f}/{scores['train_metric']:.4f}."
                )

            if self.writer:
                self._write_to_history(scores)

            if self.early_stopping:
                self._early_stopping_check(scores)
                if self.epoch - self.best_epoch > self.patience:
                    break

            if self._optuna_callback:
                self._optuna_callback(scores)

    def _write_to_history(self, scores: dict):
        """Write scores to history."""
        for key, value in scores.items():
            self.history[key].append(value)

        if self.writer:
            self.writer.add_scalar("Loss/train", scores["train_loss"], self.epoch)
            self.writer.add_scalar("Loss/val", scores["val_loss"], self.epoch)
            self.writer.add_scalar(f"{self.metric.__name__}/train", scores["train_metric"], self.epoch)
            self.writer.add_scalar(f"{self.metric.__name__}/val", scores["val_metric"], self.epoch)

    def train_epoch(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """Train model for one epoch.

        Args:
            train_loader: Train data loader.
            val_loader: Validation data loader.
        """
        self.model.train()

        train_metric = self.metric()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index, batch.batch)

            loss = self.criterion(out, batch.y.squeeze())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_metric.add(out, batch.y)

        self.model.eval()
        val_metric = self.metric()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                val_loss += self.criterion(out, batch.y.squeeze()).item()
                val_metric.add(out, batch.y)

        return {
            "train_loss": train_loss / len(train_loader),
            "train_metric": train_metric(),
            "val_loss": val_loss / len(val_loader),
            "val_metric": val_metric(),
        }

    def evaluate(self, test_loader: DataLoader, load_best=True) -> float:
        """Evaluate model on test set.

        Args:
            load_best: Load best model weights from training.
            test_loader: Test data loader.
        """
        if load_best:
            self.model.load_state_dict(torch.load(self.save_path))

        self.model.eval()
        test_metric = self.metric()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                test_loss += self.criterion(out, batch.y.squeeze()).item()
                test_metric.add(out, batch.y)

        return test_metric()
