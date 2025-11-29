# main_baseline.py
# -----------------
# 本文件是「基线单机训练」的入口脚本。
# 流程概览：
#   1. 读取配置 Config
#   2. 通过 utils_io.load_graph() 加载图结构（edge_list、用户数、物品数）
#   3. 构建 NeighborSampler 与 InteractionDataset
#   4. 构建 DataLoader（按 batch 输出 (u, pos, neg) 三元组及其子图）
#   5. 初始化 GraphSAGERecommender 模型与优化器
#   6. 调用 train_baseline() 执行多轮训练

import numpy as np
import torch

from core.config import Config
from core.loader.datasets import InteractionDataset
from core.loader.utils_io import load_graph
from core.sampler.neighbor_sampler import NeighborSampler
from core.sampler.collate_fn import collate_subgraphs
from core.models.graphsage import GraphSAGERecommender
from core.train.train_baseline import train_baseline


def main():
    """
    单机基线训练主函数。

    主要职责：
        - 加载图结构与训练数据
        - 构建邻居采样器与 DataLoader
        - 初始化 GraphSAGE 模型与优化器
        - 调用 train_baseline 完成训练循环
    """
    # 1. 加载配置（模型维度 / 采样邻居数 / batch 大小 / 学习率等）
    cfg = Config()

    # 2. 加载图结构：边列表 edge_list 以及用户/物品数量
    print("[main] 加载图结构 ...")
    edge_list, num_users, num_items = load_graph()
    print(f"[main] 图加载完成: 边数 = {edge_list.shape[0]}, 节点数 = {num_users + num_items} "
          f"(用户数 = {num_users}, 物品数 = {num_items})")

    # 3. 基于全图边列表构造串行邻居采样器
    print(f"[main] 构建 NeighborSampler, num_neighbors = {cfg.num_neighbors}")
    sampler = NeighborSampler(edge_list, cfg.num_neighbors)

    # 4. 构建交互数据集，内部会从 parquet 中读取 uid/iid
    print("[main] 加载训练数据集 data/processed/train.parquet ...")
    dataset = InteractionDataset("data/processed/train.parquet", num_items)
    print(f"[main] 训练样本数 = {len(dataset)}")

    # 5. 自定义 collate 函数：对一个 batch 内的所有 (u, pos, neg) 做子图采样
    def _collate(batch):
        """
        将原始 batch [(u, pos, neg), ...] 转换为：
            - nodes_list, maps_list: 每个样本对应的子图节点及映射
            - triples: 保留 (u, pos, neg) 三元组，供训练阶段索引 embedding
        """
        processed = []
        for u, pos, neg in batch:
            # 基于 (u, pos, neg) 三个种子节点进行多层邻居采样
            nodes, node2idx = sampler.sample([u, pos, neg])
            processed.append(((nodes, node2idx), (u, pos, neg)))
        # 统一交给 collate_subgraphs 整理为 batch 结构
        return collate_subgraphs(processed)

    # 6. 构建 DataLoader（单进程 num_workers=0 的串行版本）
    print(f"[main] 构建 DataLoader: batch_size = {cfg.batch_size}, num_workers = 0")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
    )

    # 7. 选择设备：优先使用 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] 使用设备: {device}")

    # 8. 构建 GraphSAGE 推荐模型
    print("[main] 构建 GraphSAGERecommender 模型 ...")
    model = GraphSAGERecommender(
        num_users,
        num_items,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    ).to(device)

    # 9. Adam 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # 10. 将 edge_list 转成 GNN 需要的 edge_index 形式 [2, E]
    edge_index = torch.tensor(edge_list.T, dtype=torch.long).to(device)

    # 11. 进入训练循环（见 core/train/train_baseline.py）
    print(f"[main] 开始训练: epochs = {cfg.epochs}")
    train_baseline(
        model,
        loader,
        edge_index,
        optimizer,
        device=device,
        epochs=cfg.epochs,
    )
    print("[main] 训练结束。")


if __name__ == "__main__":
    # 作为脚本直接运行时，启动基线训练
    main()
