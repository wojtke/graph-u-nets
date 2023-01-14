from typing import Callable

import torch
import torch_geometric

import metrics
import utils


def get_conv(convolution_name: str) -> Callable:
    """Get the convolution method by name."""
    CONVOLUTION_METHODS = {
        "gcn": torch_geometric.nn.conv.GCNConv,
        "gat": torch_geometric.nn.conv.GATConv,
        "sage": torch_geometric.nn.conv.SAGEConv,
        "gin": torch_geometric.nn.conv.GINConv,
    }

    if convolution_name.lower() in CONVOLUTION_METHODS:
        return CONVOLUTION_METHODS[convolution_name.lower()]
    else:
        raise ValueError(f"Convolution method '{convolution_name}' not supported.")


def get_pool(pooling_name: str) -> Callable:
    """Get the pooling method by name."""
    POOLING_METHODS = {
        "topk": torch_geometric.nn.TopKPooling,
        "sag": torch_geometric.nn.SAGPooling,
        "asap": torch_geometric.nn.ASAPooling,
    }

    if pooling_name.lower() in POOLING_METHODS:
        return POOLING_METHODS[pooling_name.lower()]
    else:
        raise ValueError(f"Pooling method '{pooling_name}' not supported.")


def get_evaluation_metric(metric_name: str) -> metrics.Metric:
    """Get the metric by name."""
    METRICS = {
        "accuracy": metrics.Accuracy,
        "auroc": metrics.AUROC,
        "rocauc": metrics.AUROC,
        "mse": metrics.MSE,
        "ap": metrics.AveragePrecision
    }

    if metric_name.lower() in METRICS:
        return METRICS[metric_name.lower()]
    else:
        raise ValueError(f"Metric '{metric_name}' not supported.")


def get_readout(readout_name: str) -> Callable:
    """Get the readout method by name."""
    READOUT_METHODS = {
        "mean": torch_geometric.nn.global_mean_pool,
        "max": torch_geometric.nn.global_max_pool,
        "add": torch_geometric.nn.global_add_pool,
        "cat": utils.readout_cat,
    }

    if readout_name.lower() in READOUT_METHODS:
        return READOUT_METHODS[readout_name.lower()]
    else:
        raise ValueError(f"Readout method '{readout_name}' not supported.")


def get_activation(activation_name: str) -> Callable:
    """Get the activation function by name."""
    ACTIVATIONS = {
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "leaky_relu": torch.nn.LeakyReLU,
    }

    if activation_name.lower() in ACTIVATIONS:
        return ACTIVATIONS[activation_name.lower()]
    else:
        raise ValueError(f"Activation function '{activation_name}' not supported.")
