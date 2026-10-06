"""Relation-aware graph attention used by the offline semantic index."""

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
        relation_text_dim: int = 384,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.heads = heads
        self.out_channels = out_channels
        self.dropout = dropout
        self.node_proj = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.relation_embedding = nn.Embedding(num_relations, heads * out_channels)
        self.relation_text_proj = nn.Linear(
            relation_text_dim, heads * out_channels, bias=False
        )
        self.att = nn.Parameter(torch.empty(1, heads, 2 * out_channels))
        self.bias = nn.Parameter(torch.zeros(heads * out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_proj.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        nn.init.xavier_uniform_(self.relation_text_proj.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_type, relation_text_embeddings):
        projected = self.node_proj(x).view(-1, self.heads, self.out_channels)
        relation = self.relation_embedding(edge_type)
        relation = relation + self.relation_text_proj(
            relation_text_embeddings[edge_type]
        )
        relation = relation.view(-1, self.heads, self.out_channels)
        out = self.propagate(
            edge_index,
            x=projected,
            relation=relation,
            size=(x.size(0), x.size(0)),
        )
        return F.elu(out.reshape(x.size(0), -1) + self.bias)

    def message(self, x_i, x_j, relation, index, ptr, size_i):
        source = x_j + relation
        attention = (torch.cat((x_i, source), dim=-1) * self.att).sum(-1)
        attention = softmax(F.leaky_relu(attention, 0.2), index, ptr, size_i)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        return source * attention.unsqueeze(-1)


class RGATEncoder(nn.Module):
    """Stacked RGAT layers for structure-enhanced node representations."""

    def __init__(self, input_dim, hidden_dim, num_relations, num_layers=4, heads=4):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.layers = nn.ModuleList(
            RGATConv(
                input_dim if layer == 0 else hidden_dim,
                hidden_dim // heads,
                num_relations,
                heads=heads,
                relation_text_dim=input_dim,
            )
            for layer in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in self.layers)

    def forward(self, x, edge_index, edge_type, relation_text_embeddings):
        for layer, norm in zip(self.layers, self.norms):
            update = layer(x, edge_index, edge_type, relation_text_embeddings)
            x = norm(x + update if x.shape == update.shape else update)
        return x
