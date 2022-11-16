import json
import os
import random

import numpy as np
import torch


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


def print_args(args):
    """Prints the argparse arguments in a nice format."""
    print("-" * 40)
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"{k:>15} : {v}")
    print("-" * 40)


def read_config(config_path):
    """Reads a config file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
