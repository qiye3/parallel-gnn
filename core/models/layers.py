# core/models/layers.py
# ---------------------
# 图神经网络基础层定义：当前仅包含简化版 GraphSAGE 卷积层。

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    """
    简化版 GraphSAGE Conv 实现。

    输入:
        x          : [N, D]，每个节点的输入特征
        edge_index : [2, E]，有向边索引 (src, dst)

    计算逻辑：
        1. 对每个节点，聚合其所有入边邻居的特征（求和）
        2. 用度数进行归一化（平均）
        3. 将自身特征 x 与聚合特征 agg 拼接
        4. 通过线性层 + ReLU 得到新的节点表示
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # GraphSAGE 通常将自身特征和邻居聚合特征拼接，因此输入维度为 2 * in_dim
        self.lin = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # edge_index 的两行分别是 src 和 dst
        src, dst = edge_index
        N = x.size(0)

        # 初始化邻居聚合结果，形状与 x 相同
        agg = torch.zeros_like(x)
        # 对于每条边 (src_i, dst_i)，将 x[src_i] 加到 agg[dst_i] 上
        agg.index_add_(0, dst, x[src])

        # 计算每个节点的入度（即被多少条边指向）
        deg = torch.bincount(dst, minlength=N).float().clamp(min=1).unsqueeze(-1)
        # 对聚合结果做平均，避免度数过大导致数值爆炸
        agg = agg / deg

        # 拼接自身特征与邻居聚合特征
        h = torch.cat([x, agg], dim=-1)
        # 线性变换 + 非线性激活
        return F.relu(self.lin(h))
