from abc import ABC, abstractmethod

import torch
from sklearn.metrics import roc_auc_score, average_precision_score


class Metric(ABC):
    """Abstract base class for metrics"""

    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def add(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Add predictions and true labels to the metric from a single batch."""
        y_true = y_true.detach().cpu()
        y_pred = y_pred.detach().cpu()

        nan_mask = torch.isnan(y_true) | torch.isnan(y_pred)
        nan_mask = nan_mask.sum(dim=1) == 0

        self.y_true.append(y_true[nan_mask])
        self.y_pred.append(y_pred[nan_mask])

        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def reset(self):
        """Reset the metric to its initial state."""
        self.y_true = []
        self.y_pred = []

    @abstractmethod
    def __call__(self):
        """Compute the metric from the predictions and true labels."""
        pass

    @classmethod
    @abstractmethod
    def direction(cls):
        """Return the direction in which to optimize metric (minimize or maximize)."""
        pass


class Accuracy(Metric):
    """Accuracy is a metric that measures the proportion of correct predictions."""

    def __call__(self):
        y_true = torch.cat(self.y_true, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)

        if y_pred.ndim == 2:
            y_pred = torch.argmax(y_pred, dim=1)

        return (y_true == y_pred).float().mean().item()

    @classmethod
    def direction(cls):
        return "maximize"


class AUROC(Metric):
    """Area under the Receiver Operating Characteristic Curve"""

    def __call__(self):
        y_true = torch.cat(self.y_true, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)

        y_pred = y_pred[:, 1] if y_pred.size(1) > 1 else y_pred

        return roc_auc_score(y_true, y_pred)

    @classmethod
    def direction(cls):
        return "maximize"


class MSE(Metric):
    """Mean Squared Error"""

    def __call__(self):
        y_true = torch.cat(self.y_true, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)

        return torch.nn.functional.mse_loss(y_pred, y_true).item()

    @classmethod
    def direction(cls):
        return "minimize"


class AveragePrecision(Metric):
    def __call__(self):
        y_true = torch.cat(self.y_true, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        return average_precision_score(y_true, y_pred)

    @classmethod
    def direction(cls):
        return "maximize"
