import torch.nn
import yaml
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset

import metrics
from mappings import get_evaluation_metric


def load_tud(dataset_name: str) -> Dataset:
    """Returns a dataset from torch_geometric.datasets.
    Also returns the task and evaluation metric associated with the dataset.

    Args:
        dataset_name (str): Name of the dataset to load.

    Returns:
        dataset (torch_geometric.data.Dataset): Loaded dataset.
    """

    dataset = TUDataset(root=f"dataset", name=dataset_name, use_node_attr=True)
    with open(f"data/{dataset_name}/config.yml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset.eval_metric = get_evaluation_metric(config["evaluation_metric"])
    dataset.task = config["task"]

    return dataset


def load_ogb(dataset_name: str):
    """Loads an OGB dataset.

    Args:
        dataset_name (str): Name of the dataset to load.
    """
    dataset = PygGraphPropPredDataset(name=dataset_name)
    dataset.eval_metric = metrics.wrap_ogb_eval_metric(dataset_name)

    return dataset


def get_out_channels(dataset: Dataset) -> int:
    """Returns the number of output channels for a dataset.

    Args:
        dataset (torch_geometric.data.Dataset): Dataset to get output channels for.

    Returns:
        out_channels (int): Number of output channels.
    """
    if dataset.task_type == "binary classification":
        return dataset.num_tasks
    elif dataset.task_type == "multi-class classification":
        return dataset.num_classes
    else:
        raise ValueError(f"Invalid task: {dataset.task}")


def get_criterion(dataset: Dataset):
    """Returns the loss function for a dataset.

    Args:
        dataset (torch_geometric.data.Dataset): Dataset to get loss function for.
    """
    if dataset.task_type == "binary classification":
        return torch.nn.BCEWithLogitsLoss()
    elif dataset.task_type == "multi-class classification":
        return torch.nn.CrossEntropyLoss()
    elif dataset.task_type == "regression":
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"Invalid task: {dataset.task_type}")
