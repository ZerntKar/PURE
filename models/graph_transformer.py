import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool


class GraphTransformerLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        heads: int = 2,
        dropout: float = 0.1,
        edge_dim: int = 64,
    ):
        super().__init__()
        self.conv = TransformerConv(
            in_channels,
            out_channels // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            beta=True,
        )
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Linear(out_channels * 4, out_channels),
        )
        self.dropout = nn.Dropout(dropout)
        self.rel_emb = nn.Embedding(num_relations + 1, edge_dim, padding_idx=0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        edge_attr = self.rel_emb(edge_type)
        h_attn = self.conv(self.norm1(x), edge_index, edge_attr)
        h = x + self.dropout(h_attn)
        h_ffn = self.ffn(self.norm2(h))
        h = h + self.dropout(h_ffn)
        return h


class GraphTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_relations: int,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_channels, hidden_channels, num_relations,
                heads=heads, dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_type)
        return global_mean_pool(h, batch)
