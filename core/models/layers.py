# core/models/layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    """
    简化版 GraphSAGE Conv
    x: [N, D]
    edge_index: [2, E]  (src, dst)
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, edge_index):
        src, dst = edge_index
        N = x.size(0)

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])

        deg = torch.bincount(dst, minlength=N).float().clamp(min=1).unsqueeze(-1)
        agg = agg / deg

        h = torch.cat([x, agg], dim=-1)
        return F.relu(self.lin(h))
