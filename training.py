import numpy as np
import torch
from torch_geometric.loader import DataLoader

from callbacks import History, ProgressBar
from metrics import Accuracy, Metric


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        metric: Metric = Accuracy,
    ):
        """Initialize trainer.

        Args:
            model: Model to train.
            optimizer: Optimizer to use.
            criterion: Loss function.
            metric: Metric to use for evaluation. (default: Accuracy)
        """
        self.history = None
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric
        self.epoch = 0

        self.stop_training_flag = False

        self.callbacks = {}
        self.add_callbacks(History(), ProgressBar())

    def add_callbacks(self, *callbacks):
        """Add callback to trainer.

        Args:
            *callbacks: Callback to add.
        """
        for callback in callbacks:
            self.callbacks.update({callback.__class__: callback})
            callback.setup(self)

    def run_callbacks(self, logs: dict):
        """Run callbacks after each epoch."""
        for callback in self.callbacks.values():
            callback(self, logs)
            callback(logs)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 1000):
        """Train model.

        Args:
            train_loader: Train data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs to train; if early stopping is enabled, this is the maximum number of epochs.
        """
        while self.epoch < epochs and not self.stop_training_flag:
            self.epoch += 1
            logs = self.train_epoch(train_loader)
            with torch.no_grad():
                val_logs = self.eval_epoch(val_loader)
            logs.update(val_logs)
            self.run_callbacks(logs)

    def train_epoch(self, loader: DataLoader) -> dict:
        """Train model for one epoch.

        Args:
            loader: Train data loader.
        """
        self.model.train()
        train_metric = self.metric()
        train_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(out, batch.y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_metric.add(out, batch.y)

        return {"train_loss": train_loss / len(loader), "train_metric": train_metric.compute()}

    def eval_epoch(self, loader: DataLoader) -> dict:
        """Evaluate model on data loader.

        Args:
            loader: Data loader.
        """
        self.model.eval()
        metric = self.metric()
        loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss += self.criterion(out, batch.y).item()
            metric.add(out, batch.y)

        return {"val_loss": loss / len(loader), "val_metric": metric.compute()}




