# core/models/graphsage.py

import torch
import torch.nn as nn
from .layers import GraphSAGELayer


class GraphSAGERecommender(nn.Module):
    """
    user-item 二部图的 GraphSAGE 推荐模型
    """

    def __init__(self, num_users, num_items,
                 hidden_dim=64, num_layers=2):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items

        self.emb = nn.Embedding(self.num_nodes, hidden_dim)

        self.layers = nn.ModuleList(
            [GraphSAGELayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

    def forward_full(self, edge_index):
        device = edge_index.device
        nodes = torch.arange(self.num_nodes, device=device, dtype=torch.long)
        x = self.emb(nodes)
        for layer in self.layers:
            x = layer(x, edge_index)
        return x  # [num_nodes, dim]

    def score(self, u_emb, i_emb):
        return (u_emb * i_emb).sum(-1)
