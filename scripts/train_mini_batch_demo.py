# scripts/train_mini_batch_demo.py
# --------------------------------
# Mini-batch GraphSAGE 训练演示脚本。
# 与 baseline 训练（main_baseline.py）不同，该脚本使用真正的 mini-batch 训练方式：
#   - 每个 batch 构建一个子图，在子图上执行 GNN 前向传播
#   - 避免了 baseline 中"先全图前向、再基于 embedding 打分"的两阶段策略
#   - 更适合大规模图训练，显存占用更可控

import os
import sys
import torch
import numpy as np

# 将项目根目录添加到 Python 路径，以便导入 core 模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config import Config
from core.loader.datasets import InteractionDataset
from core.loader.utils_io import load_graph
from core.sampler.neighbor_sampler import NeighborSampler
from core.models.graphsage import GraphSAGERecommender
from core.train.train_mini_batch import train_mini_batch_gnn


def main():
    """
    Mini-batch GraphSAGE 训练主函数。

    训练流程：
        1. 加载图结构和配置
        2. 创建邻居采样器和数据集
        3. 从完整数据集中随机采样子集（用于快速实验）
        4. 构建模型和优化器
        5. 调用 mini-batch 训练函数进行训练
    """
    # 加载配置参数
    cfg = Config()

    # 加载图结构：边列表、用户数、物品数
    edge_list, num_users, num_items = load_graph()
    # 创建邻居采样器，用于为每个 batch 构建子图
    sampler = NeighborSampler(edge_list, cfg.num_neighbors)

    # 加载完整的训练数据集
    # 注意：这里使用默认的 collate_fn，返回 (u_batch, pos_batch, neg_batch) 三个张量
    # 而不是 baseline 中使用的自定义 collate_fn（返回子图结构）
    full_dataset = InteractionDataset("data/processed/train.parquet", num_items)

    # 为了快速实验和调试，从完整数据集中随机采样一部分样本
    # 实际训练时可以注释掉这部分，直接使用 full_dataset
    subset_size = 10_000
    indices = np.random.choice(len(full_dataset), size=subset_size, replace=False)
    subset = torch.utils.data.Subset(full_dataset, indices)

    # 构建 DataLoader，使用默认的 collate_fn
    # 注意：这里 num_workers=0，如果需要多进程采样可以改为 >0
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0
    )

    # 选择训练设备（优先使用 GPU）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[mini-batch-demo] 使用设备:", device)

    # 构建 GraphSAGE 推荐模型
    model = GraphSAGERecommender(
        num_users,
        num_items,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    ).to(device)

    # 创建优化器（Adam）
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # 执行 mini-batch 训练
    # 训练过程中，每个 batch 会：
    #   1. 从 batch 中提取 (u, pos, neg) 三元组
    #   2. 以这些节点为种子，使用 NeighborSampler 构建子图
    #   3. 在子图上执行 GNN 前向传播（forward_subgraph）
    #   4. 计算损失并反向传播更新参数
    train_mini_batch_gnn(
        model,
        loader,
        sampler,
        optimizer,
        device=device,
        epochs=2,
    )


if __name__ == "__main__":
    main()
