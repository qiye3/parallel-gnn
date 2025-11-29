# core/models/gat.py
# ------------------
# 图注意力网络（GAT）层及其推荐模型封装。

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    单头 GAT 注意力层。

    计算流程：
        1. 线性变换节点特征 h = W x
        2. 对每条边 (i, j) 计算注意力打分 e_ij = LeakyReLU(a [h_i || h_j])
        3. 对每个节点 j 的入边归一化注意力权重 α_ij
        4. 聚合邻居特征：h'_j = Σ_i α_ij h_i
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        # 特征线性变换
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # 注意力打分网络，输入为拼接后的 [h_i, h_j]
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 边的两端节点索引
        src, dst = edge_index
        # 线性变换后的特征
        h = self.W(x)

        # 为每条边组合 (h_i, h_j) 用于计算注意力打分
        h_cat = torch.cat([h[src], h[dst]], dim=-1)
        e = self.leaky_relu(self.a(h_cat)).squeeze()

        # 对每个节点 j 的入边做 softmax 归一化（此处用手写形式实现）
        att = torch.exp(e)
        att_sum = torch.zeros(h.size(0), device=h.device)
        att_sum.index_add_(0, dst, att)
        att_norm = att / (att_sum[dst] + 1e-12)

        # 按注意力权重聚合邻居特征
        out = torch.zeros_like(h)
        out.index_add_(0, dst, h[src] * att_norm.unsqueeze(-1))
        return F.elu(out)


class GATRecommender(nn.Module):
    """
    基于 GAT 的推荐模型，与 GraphSAGERecommender 结构类似。
    """

    def __init__(self, num_users: int, num_items: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.num_nodes = num_users + num_items

        # 节点嵌入
        self.emb = nn.Embedding(self.num_nodes, hidden_dim)
        # 多层 GAT
        self.layers = nn.ModuleList(
            [GATLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

    def forward_full(self, edge_index: torch.Tensor) -> torch.Tensor:
        """在整图上做一次 GAT 前向传播，返回所有节点的表示。"""
        nodes = torch.arange(self.num_nodes, device=edge_index.device)
        x = self.emb(nodes)
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

    def score(self, u_emb: torch.Tensor, i_emb: torch.Tensor) -> torch.Tensor:
        """与 GraphSAGERecommender 一致：使用内积作为打分函数。"""
        return (u_emb * i_emb).sum(-1)
