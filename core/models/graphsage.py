# core/models/graphsage.py
# -------------------------
# GraphSAGE 推荐模型：基于用户-物品二部图的图神经网络推荐系统核心模型。
# 支持整图前向传播（用于 baseline 训练和全图推理）和子图前向传播（用于 mini-batch 训练）。

import torch
import torch.nn as nn
from .layers import GraphSAGELayer


class GraphSAGERecommender(nn.Module):
    """
    基于 GraphSAGE 的推荐模型，用于用户-物品二部图上的推荐任务。

    设计特点：
        - 将用户和物品统一视为图中的节点，使用共享的嵌入矩阵
        - 通过多层 GraphSAGE 卷积捕获高阶邻居信息
        - 支持两种前向模式：整图前向（baseline/推理）和子图前向（mini-batch 训练）
        - 使用内积作为用户-物品相似度打分函数

    主要接口：
        - forward_full: 在整张图上执行 GNN 前向传播，返回所有节点的嵌入
        - forward_subgraph: 在采样的子图上执行 GNN 前向传播，用于 mini-batch 训练
        - score: 计算用户嵌入与物品嵌入的相似度得分
    """

    def __init__(self, num_users, num_items,
                 hidden_dim=64, num_layers=2):
        """
        初始化 GraphSAGE 推荐模型。

        参数:
            num_users   : 用户总数
            num_items   : 物品总数
            hidden_dim  : 节点嵌入的隐藏维度（默认 64）
            num_layers  : GraphSAGE 卷积层数（默认 2 层）
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        # 总节点数 = 用户数 + 物品数（用户和物品共享同一节点空间）
        self.num_nodes = num_users + num_items

        # 为所有节点（用户+物品）创建统一的嵌入矩阵
        # 用户节点 ID: [0, num_users)，物品节点 ID: [num_users, num_users + num_items)
        self.emb = nn.Embedding(self.num_nodes, hidden_dim)

        # 堆叠多层 GraphSAGE 卷积层，实现多跳邻居信息聚合
        self.layers = nn.ModuleList(
            [GraphSAGELayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

    def forward_full(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        整图前向传播：在完整的用户-物品二部图上执行一次 GNN 前向传播。

        该方法用于：
            - baseline 训练：每个 epoch 开始时计算所有节点的嵌入，然后基于这些嵌入计算损失
            - 全图推理：为所有用户和物品生成最终的节点表示，用于推荐打分

        参数:
            edge_index: [2, E] 形状的边索引张量，第一行为源节点，第二行为目标节点

        返回:
            torch.Tensor: [num_nodes, hidden_dim] 形状的节点嵌入矩阵
        """
        device = edge_index.device
        # 创建所有节点的索引 [0, 1, 2, ..., num_nodes-1]
        nodes = torch.arange(self.num_nodes, device=device, dtype=torch.long)
        # 获取所有节点的初始嵌入
        x = self.emb(nodes)
        # 逐层执行 GraphSAGE 卷积，每层聚合邻居信息并更新节点表示
        for layer in self.layers:
            x = layer(x, edge_index)
        return x  # [num_nodes, hidden_dim]

    def forward_subgraph(
        self,
        global_node_ids: torch.Tensor,
        edge_index_sub: torch.Tensor,
    ) -> torch.Tensor:
        """
        子图前向传播：在采样的子图上执行 GNN 前向传播，用于 mini-batch 训练。

        该方法用于：
            - mini-batch 训练：每个 batch 只对相关的子图节点进行前向传播，节省计算和显存

        参数:
            global_node_ids: [N_sub] 形状的张量，包含子图中所有节点的全局 ID
            edge_index_sub : [2, E_sub] 形状的边索引，使用子图内的局部索引 [0..N_sub-1]

        返回:
            torch.Tensor: [N_sub, hidden_dim] 形状的子图节点嵌入矩阵
        """
        # 根据全局节点 ID 获取对应的嵌入向量
        x = self.emb(global_node_ids)  # [N_sub, hidden_dim]
        # 在子图上逐层执行 GraphSAGE 卷积
        for layer in self.layers:
            x = layer(x, edge_index_sub)
        return x  # 子图节点 embedding

    def score(self, u_emb: torch.Tensor, i_emb: torch.Tensor) -> torch.Tensor:
        """
        计算用户嵌入与物品嵌入的相似度得分（用于推荐排序）。

        使用简单的内积（点积）作为相似度度量：
            score = sum(u_emb * i_emb)

        参数:
            u_emb: 用户嵌入向量，形状为 [hidden_dim] 或 [batch_size, hidden_dim]
            i_emb: 物品嵌入向量，形状与 u_emb 相同

        返回:
            torch.Tensor: 相似度得分，形状为标量或 [batch_size]
        """
        return (u_emb * i_emb).sum(-1)
