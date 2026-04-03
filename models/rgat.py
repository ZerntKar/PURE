import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class RGATConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.num_relations = num_relations
        self.heads         = heads
        self.dropout       = dropout
        self.concat        = concat

        self.W_rel = nn.Parameter(
            torch.empty(num_relations, in_channels, out_channels * heads)
        )
        self.att = nn.Parameter(torch.empty(1, heads, 2 * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(
            self.W_rel.view(self.num_relations, -1).unsqueeze(0)
            .expand(1, -1, -1).squeeze(0)
        )
        nn.init.zeros_(self.att)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:

        w_per_edge    = self.W_rel[edge_type]
        x_src         = x[edge_index[0]]
        x_transformed = torch.einsum("ei,eio->eo", x_src, w_per_edge)
        x_transformed = x_transformed.view(-1, self.heads, self.out_channels)

        W_mean = self.W_rel.mean(0)
        x_all  = (x @ W_mean).view(-1, self.heads, self.out_channels)

        out = self.propagate(
            edge_index,
            x=x_all,
            x_src_transformed=x_transformed,
            size=(x.size(0), x.size(0)),
        )

        if self.concat:
            return F.elu(out.view(-1, self.heads * self.out_channels))
        else:
            return F.elu(out.mean(dim=1))

    def message(
        self,
        x_i: torch.Tensor,
        x_src_transformed: torch.Tensor,
        index: torch.Tensor,
        ptr,
        size_i,
    ) -> torch.Tensor:
        alpha = (torch.cat([x_i, x_src_transformed], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_src_transformed * alpha.unsqueeze(-1)
