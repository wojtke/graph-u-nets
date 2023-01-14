from abc import ABC, abstractmethod

import numpy as np
import optuna
import torch
from torch.utils.tensorboard import SummaryWriter


class Callback(ABC):
    def __init__(self):
        pass

    def setup(self, trainer):
        self.trainer = trainer

    @abstractmethod
    def __call__(self, logs: dict):
        pass


class EarlyStopping(Callback):
    def __init__(self, patience=10, min_epochs=25, save_path=None, monitor="val_metric", direction="maximize"):
        super().__init__()
        self.patience = patience
        self.min_epochs = min_epochs
        self.save_path = save_path
        self.best_epoch = 0
        self.best_value = np.inf if direction == "minimize" else -np.inf
        self.monitor = monitor
        self.direction = direction

    def __call__(self, logs: dict):
        if self.direction == "minimize" and logs[self.monitor] < self.best_value \
                or self.direction == "maximize" and logs[self.monitor] > self.best_value:
            self.best_value = logs[self.monitor]
            self.best_epoch = self.trainer.epoch
            if self.save_path is not None:
                torch.save(self.trainer.model.state_dict(), self.save_path)
        elif self.trainer.epoch - self.best_epoch > self.patience and self.trainer.epoch > self.min_epochs:
            self.trainer.stop_training_flag = True


class OptunaPruning(Callback):
    def __init__(self, trial, monitor="val_metric"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def __call__(self, logs: dict):
        self.trial.report(logs[self.monitor], self.trainer.epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


class TensorBoardWriter(Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.writer = SummaryWriter(log_dir)

    def __call__(self, logs: dict):
        for key, value in logs.items():
            self.writer.add_scalar(key, value, self.trainer.epoch)

    def close(self):
        self.writer.close()


class ProgressBar(Callback):
    def __init__(self, monitor="val_metric"):
        super().__init__()
        self.monitor = monitor

    def __call__(self, logs: dict):
        print(f"Epoch {self.trainer.epoch}: {self.monitor} = {logs[self.monitor]}")


class History(Callback):
    def __init__(self, metric=None):
        super().__init__()
        self.history = {}
        self.metric = metric

    def __getstate__(self):
        return self.history

    def setup(self, trainer):
        super().setup(trainer)
        self.trainer.history = self

    def __call__(self, logs: dict):
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def get_best(self):
        if self.metric is None:
            raise ValueError("Metric not set.")
        return max(self.history["val_metric"]) if self.metric.direction == "maximize" \
            else min(self.history["val_metric"])



