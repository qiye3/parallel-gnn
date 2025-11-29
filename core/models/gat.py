# core/models/gat.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        src, dst = edge_index
        h = self.W(x)

        h_cat = torch.cat([h[src], h[dst]], dim=-1)
        e = self.leaky_relu(self.a(h_cat)).squeeze()

        att = torch.exp(e)
        att_sum = torch.zeros(h.size(0), device=h.device)
        att_sum.index_add_(0, dst, att)
        att_norm = att / (att_sum[dst] + 1e-12)

        out = torch.zeros_like(h)
        out.index_add_(0, dst, h[src] * att_norm.unsqueeze(-1))
        return F.elu(out)


class GATRecommender(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim=64, num_layers=2):
        super().__init__()
        self.num_nodes = num_users + num_items

        self.emb = nn.Embedding(self.num_nodes, hidden_dim)
        self.layers = nn.ModuleList(
            [GATLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

    def forward_full(self, edge_index):
        nodes = torch.arange(self.num_nodes, device=edge_index.device)
        x = self.emb(nodes)
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

    def score(self, u_emb, i_emb):
        return (u_emb * i_emb).sum(-1)
