from typing import Callable, List, Union

import torch
import torch.nn.functional as F
import torch_geometric
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    sort_edge_index,
)
from torch_geometric.utils.repeat import repeat
from torch_sparse import spspmm


class GNN(torch.nn.Module):
    """Graph neural network consisting of a u-net, readout and MLP layers."""

    def __init__(self, g_unet: torch.nn.Module, readout: str, out: torch.nn.Module):
        """Initializes the GNN.

        Args:
            g_unet (torch.nn.Module): Graph u-net module.
            readout (str): Readout function. One of "add", "mean", "max".
            out (torch.nn.Module): Final MLP layers for classification.
        """
        super().__init__()
        self.g_unet = g_unet
        self.readout = {
            "mean": torch_geometric.nn.global_mean_pool,
            "max": torch_geometric.nn.global_max_pool,
            "add": torch_geometric.nn.global_add_pool,
        }[readout]
        self.out = out

    def forward(self, x, edge_index, batch):
        x = self.g_unet(x, edge_index, batch)
        x = self.readout(x, batch)
        x = self.out(x)
        return x

    def reset_parameters(self):
        self.g_unet.reset_parameters()
        self.out.reset_parameters()


# TODO add dropout
class GraphUNet(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        hidden_channels (int): Size of each hidden sample in the U-Net.
        depth (int): The depth of the U-Net architecture.
        mlp_layers (int, optional): The number of layers in mlp
        mlp_hidden_channels (int, optional): The hidden dimension of mlp


        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
        pool : The pooling layer to use.
            (default: :class:`torch_geometric.nn.TopKPooling`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        depth: int = 3,
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = True,
        act: Union[Callable, str] = "relu",
        pool: Union[Callable, str] = "TopKPooling",
    ) -> None:
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.sum_res = sum_res

        self.act = (
            act
            if isinstance(act, Callable)
            else {
                "relu": F.relu,
                "elu": F.elu,
                "leakyrelu": F.leaky_relu,
            }[act.lower()]
        )

        self.pool = (
            pool
            if isinstance(act, Callable)
            else {
                "topkpooling": torch_geometric.nn.TopKPooling,
                "sagpooling": torch_geometric.nn.SAGPooling,
            }[pool.lower()]
        )

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        for i in range(depth):
            self.pools.append(self.pool(int(channels), self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved=True))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GCNConv(in_channels, channels, improved=True))
        self.up_convs.append(GCNConv(in_channels, out_channels, improved=True))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch: OptTensor = None) -> Tensor:
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(
                edge_index, edge_weight, x.size(0)
            )
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch
            )

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

        return x

    def augment_adj(
        self, edge_index: Tensor, edge_weight: Tensor, num_nodes: int
    ) -> PairTensor:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, num_nodes=num_nodes
        )
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
