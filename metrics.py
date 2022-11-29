import torch
from abc import ABC, abstractmethod
from sklearn.metrics import roc_auc_score


class Metric(ABC):
    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def add(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self.y_true.append(y_true.cpu())
        self.y_pred.append(y_pred.cpu())

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def __call__(self):
        pass

    @classmethod
    @abstractmethod
    def direction(cls):
        pass


class Accuracy(Metric):
    def __call__(self):
        y_true = torch.cat(self.y_true, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        return (y_true == y_pred).float().mean().item()

    @classmethod
    def direction(cls):
        return "maximize"


class AUROC(Metric):
    def __call__(self):
        y_true = torch.cat(self.y_true, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        return roc_auc_score(y_true, y_pred)

    @classmethod
    def direction(cls):
        return "maximize"


class MSE(Metric):
    def __call__(self):
        y_true = torch.cat(self.y_true, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        return torch.nn.functional.mse_loss(y_pred, y_true).item()

    @classmethod
    def direction(cls):
        return "minimize"
