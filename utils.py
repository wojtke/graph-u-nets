import os
import random

import numpy as np
import torch
import torch_geometric


def set_reproducibility(seed):
    """Sets seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir_if_not_exists(path):
    """Creates a directory if it does not already exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def readout_cat(x: torch.Tensor, batch) -> torch.Tensor:
    """Concatenates the readout of the three types of global pooling."""
    a = torch_geometric.nn.global_mean_pool(x, batch)
    b = torch_geometric.nn.global_max_pool(x, batch)
    c = torch_geometric.nn.global_add_pool(x, batch)
    return torch.cat([a, b, c], dim=1)


def print_args(args):
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"{k:>20} : {v}")
    print()


def get_device(device=None) -> torch.device:
    """Get the device."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch.device(device)
