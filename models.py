from typing import Callable, List, Union

import torch
import torch.nn.functional as F
import torch_geometric
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.typing import PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    sort_edge_index,
)
from torch_geometric.utils.repeat import repeat
from torch_sparse import spspmm

from hyperparams import Hyperparams


class GNN(torch.nn.Module):
    """Graph neural network consisting of a graph U-Net, readout and MLP layers."""

    def __init__(self, in_channels: int, out_channels: int, hyperparams: Hyperparams):
        """Initializes the GNN.

        Args:
            in_channels: Number of input features.
            out_channels: Number of output features.
            hyperparams: Hyperparameters dataclass.
        """
        super().__init__()

        self.g_unet = GraphUNet(
            in_channels=in_channels,
            hidden_channels=hyperparams.hidden_channels,
            out_channels=hyperparams.hidden_channels,
            depth=hyperparams.depth,
            pool_ratios=hyperparams.pool_ratios,
            dropout=hyperparams.dropout,
            conv=hyperparams.conv,
            pool=hyperparams.pool,
        )
        self.readout = hyperparams.readout
        if self.readout.__name__ == "readout_cat":
            channels = 3 * self.g_unet.out_channels
        else:
            channels = self.g_unet.out_channels
        self.out = MLP(
            in_channels=channels,
            hidden_channels=hyperparams.hidden_channels,
            out_channels=out_channels,
            num_layers=1,
            act=hyperparams.act,
        )

    def forward(self, x, edge_index, batch=None):
        if x is None:
            x = torch.ones((batch.size(0), self.g_unet.in_channels), device=edge_index.device)
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        x = x.float()

        x = self.g_unet(x, edge_index, batch)
        x = self.readout(x, batch)
        x = self.out(x)
        return x

    def reset_parameters(self):
        self.g_unet.reset_parameters()
        self.out.reset_parameters()


class MLP(torch.nn.Sequential):
    """Multi-layer perceptron."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: int,
            num_layers: int,
            dropout: int = 0.0,
            act: Callable = torch.nn.ReLU,
    ):
        """Initializes the MLP.

        Args:
            in_channels: Number of input features.
            out_channels: Number of output features.
            hidden_channels: Number of hidden features.
            num_layers: Number of layers.
            dropout: Dropout probability after each activation.
            act: Activation function.
        """
        super().__init__()

        if num_layers == 1:
            self.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.append(torch.nn.Linear(in_channels, hidden_channels))
            self.append(act())
            self.append(torch.nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                self.append(torch.nn.Linear(hidden_channels, hidden_channels))
                self.append(act())
                self.append(torch.nn.Dropout(dropout))
            self.append(torch.nn.Linear(hidden_channels, out_channels))


class GraphUNet(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        hidden_channels (int): Size of each hidden sample in the U-Net.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float]): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (Callable): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
        pool (Callable) : The pooling layer to use.

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: int,
            depth: int = 3,
            pool_ratios: Union[float, List[float]] = 0.5,
            dropout: float = 0.0,
            sum_res: bool = True,
            act: Callable = F.elu,
            conv: Callable = GCNConv,
            pool: Callable = torch_geometric.nn.TopKPooling,
    ) -> None:
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.sum_res = sum_res
        self.act = act

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        for i in range(depth):
            self.pools.append(pool(int(channels), self.pool_ratios[i]))
            self.down_convs.append(conv(channels, channels, improved=True))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(conv(in_channels, channels, improved=True))
        self.up_convs.append(conv(in_channels, out_channels, improved=True))

        self.dropout = torch.nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            x = self.dropout(x)

            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))

            x, edge_index, edge_weight, batch, perm = self.pools[i - 1](x, edge_index, edge_weight, batch)[:5]

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x
            if self.dropout.p > 0:
                x = self.dropout(x)

        return x

    def augment_adj(self, edge_index: Tensor, edge_weight: Tensor, num_nodes: int) -> PairTensor:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight, num_nodes)
        edge_index, edge_weight = spspmm(
            edge_index,
            edge_weight,
            edge_index,
            edge_weight,
            num_nodes,
            num_nodes,
            num_nodes,
        )
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.hidden_channels}, {self.out_channels}, "
            f"depth={self.depth}, pool_ratios={self.pool_ratios})"
        )
