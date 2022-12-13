import os
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset

import metrics
from mappings import get_evaluation_metric


def load_dataset(dataset_name: str) -> Dataset:
    """Returns a dataset from torch_geometric.datasets.
    Also returns the task and evaluation metric associated with the dataset.

    Args:
        dataset_name (str): Name of the dataset to load.

    Returns:
        dataset (torch_geometric.data.Dataset): Loaded dataset.
        task (str): Task to be performed (classification or regression).
        evaluation_metric (metrics.Metric): Metric to be used for model evaluation.
    """
    if dataset_name != "":
        dataset = TUDataset(root="data", name=dataset_name)
    elif dataset_name in ["HIV"]:
        dataset = TUDataset(root="data", name=dataset_name)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    return dataset


def load_dataset_artifacts(dataset_name: str) -> Tuple[str, metrics.Metric]:
    with open(f"data/{dataset_name}/config.yml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    task = config["task"]
    metric = get_evaluation_metric(config["evaluation_metric"])
    return task, metric


def load_splits(dataset_name: str):
    """Loads saved train and test splits for a dataset.

    Note: Need to run split_dataset.py first to generate the splits.

    Args:
        dataset_name (str): Name of the dataset to load splits for.
    """
    splits = []
    for file_name in os.listdir(f"data/{dataset_name}/splits"):
        if file_name.startswith("train"):
            with open(f"data/{dataset_name}/splits/{file_name}", "r") as f:
                train_idx = pd.read_csv(f, header=None).values.flatten()
            with open(f"data/{dataset_name}/splits/{file_name.replace('train', 'test')}", "r") as f:
                test_idx = pd.read_csv(f, header=None).values.flatten()
            splits.append((list(train_idx), list(test_idx)))
    return splits


def split_indices(dataset: Dataset, test_size=0.1, indices=None, method=None):
    """Returns train and test indices for a dataset.

    Args:
        dataset (torch_geometric.data.Dataset): Dataset to split into train and test.
        test_size (float): Proportion of the dataset to use for testing (default: 0.2).
        indices (list): Indices to use for splitting (default: None).
    """
    if indices is None:
        indices = np.arange(dataset.len())

    if method == "stratified":
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, stratify=[data.y.item() for data in dataset[indices]]
        )
    else:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
        )

    return list(train_idx), list(test_idx)
