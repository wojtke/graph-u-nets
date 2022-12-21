from dataclasses import dataclass
from typing import Callable

import optuna
import yaml

from mappings import get_readout, get_conv, get_pool, get_activation


@dataclass
class Hyperparams:
    """Dataclass used to store hyperparameters for the model and its training."""

    hidden_channels: int
    depth: int
    pool_ratios: list
    dropout: float
    act: Callable
    conv: Callable
    pool: Callable
    readout: Callable

    batch_size: int
    lr: float
    weight_decay: float

    min_epochs: int = 100
    patience: int = 100
    max_epochs: int = 1000

    def save(self, path):
        """Saves to a yaml file."""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)

    @classmethod
    def load(cls, path, **params):
        """Loads from a yaml file.

        Args:
            path: Path to the yaml file.
            **params: Additional params to override or supplement the ones in the yaml file.
        """
        with open(path, "r") as f:
            loaded_params = yaml.load(f, Loader=yaml.FullLoader)
        loaded_params.update(params)
        return cls(**loaded_params)

    def __post_init__(self):
        """Maps params passed as str to callables."""
        if isinstance(self.readout, str):
            self.readout = get_readout(self.readout)
        if isinstance(self.conv, str):
            self.conv = get_conv(self.conv)
        if isinstance(self.pool, str):
            self.pool = get_pool(self.pool)
        if isinstance(self.act, str):
            self.act = get_activation(self.act)

    def __repr__(self):
        return "\n".join([f"{k:>15} : {v}" for k, v in self.__dict__.items()])


@dataclass
class HyperparamsSpace(Hyperparams):
    """Dataclass used to store hyperparameters for the model and its training."""

    def pick(self, trial: optuna.Trial, **params):
        """Pick hyperparameters from the space for optuna trial.

        Args:
            trial: optuna trial.
            **params: additional params to override the space.
        """
        params = {k: self._pick_one(trial, k) for k in Hyperparams.__annotations__ if k not in params}
        return Hyperparams(**params)

    def _pick_one(self, trial: optuna.Trial, param_name: str):
        """Pick one hyperparameter from the space for optuna trial.
        Uses optuna's suggest_<type> methods to pick the hyperparameter.
        """
        param = self.__getattribute__(param_name)
        if isinstance(param, dict):
            suggest_func = getattr(trial, f"suggest_{param['type']}")
            return suggest_func(param_name, **{k: v for k, v in param.items() if k != "type"})
        elif param is None:
            raise ValueError(f"Hyperparam '{param_name}' not defined.")
        else:
            return param

    def __repr__(self):
        return "\n".join([f"{k:>15} : {v}" for k, v in self.__dict__.items()])
