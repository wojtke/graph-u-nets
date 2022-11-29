from dataclasses import dataclass
from typing import Callable

import optuna
import yaml


@dataclass
class Hyperparams:
    in_channels: int
    hidden_channels: int
    out_channels: int
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

    min_epochs: int = 50
    patience: int = 50
    max_epochs: int = 1000

    def save(self, path):
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            return cls(**yaml.load(f, Loader=yaml.FullLoader))


@dataclass
class HyperparamsSpace(Hyperparams):
    def pick(self, trial: optuna.Trial, **params):
        params = {k: self._pick_one(trial, k) for k in Hyperparams.__annotations__ if k not in params}

        return Hyperparams(**params)

    def _pick_one(self, trial: optuna.Trial, param_name: str):
        param = self.__getattribute__(param_name)
        if isinstance(param, (int, float, str, bool, list, tuple)):
            return param
        elif isinstance(param, dict):
            suggest_func = getattr(trial, f"suggest_{param['type']}")
            return suggest_func(param_name, **{k: v for k, v in param.items() if k != "type"})
        else:
            raise ValueError(f"Hyperparam {param_name} not defined properly.")
