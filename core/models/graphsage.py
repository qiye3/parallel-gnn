# core/models/graphsage.py
# ------------------------
# 基于 GraphSAGE 的推荐模型实现，适用于 user-item 二部图。

import torch
import torch.nn as nn
from .layers import GraphSAGELayer


class GraphSAGERecommender(nn.Module):
    """
    user-item 二部图的 GraphSAGE 推荐模型。

    设计要点：
        - 将用户与物品统一视为图上的节点：
              [0, num_users)         为用户节点
              [num_users, num_nodes) 为物品节点
        - 使用 nn.Embedding 存储所有节点的初始表示
        - 通过多层 GraphSAGELayer 聚合邻居信息
        - 评分函数 score 采用简单的内积
    """

    def __init__(self, num_users: int, num_items: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        # 总节点数 = 用户数 + 物品数
        self.num_nodes = num_users + num_items

        # 所有节点的可学习嵌入矩阵 [num_nodes, hidden_dim]
        self.emb = nn.Embedding(self.num_nodes, hidden_dim)

        # 叠加多层 GraphSAGE 卷积层
        self.layers = nn.ModuleList(
            [GraphSAGELayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

    def forward_full(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        在整张图上做一次完整的前向传播，得到所有节点的最终表示。

        参数:
            edge_index: 形状为 [2, E] 的长整型张量，第一行是 src，第二行是 dst

        返回:
            x: 形状为 [num_nodes, hidden_dim] 的节点嵌入矩阵
        """
        device = edge_index.device
        # 构造 [0, num_nodes) 的节点 ID 序列
        nodes = torch.arange(self.num_nodes, device=device, dtype=torch.long)
        # 初始嵌入
        x = self.emb(nodes)
        # 逐层执行 GraphSAGE 聚合
        for layer in self.layers:
            x = layer(x, edge_index)
        # 最终的节点表示，可被下游任务直接索引
        return x  # [num_nodes, hidden_dim]

    def score(self, u_emb: torch.Tensor, i_emb: torch.Tensor) -> torch.Tensor:
        """
        用户嵌入与物品嵌入的打分函数。

        这里采用最简单的内积形式：
            score(u, i) = <u_emb, i_emb>
        """
        return (u_emb * i_emb).sum(-1)
