"""
scripts/quick_train.py
----------------------
一个用于「快速训练 / 烟雾测试」的小脚本：
    - 只随机采样训练集的一小部分样本
    - 只跑少量 epoch
    - 方便检查代码是否能正常跑通、观察日志输出

使用方式（在项目根目录下）：
    python scripts/quick_train.py
"""

import os
import sys
import numpy as np
import torch


# ---------- 确保可以导入项目根目录下的 core 包 ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config import Config
from core.loader.datasets import InteractionDataset
from core.loader.utils_io import load_graph
from core.sampler.neighbor_sampler import NeighborSampler
from core.sampler.collate_fn import collate_subgraphs
from core.models.graphsage import GraphSAGERecommender
from core.train.train_baseline import train_baseline


def main():
    """
    快速训练主函数：
        - 从完整训练集随机采样一小部分样本
        - 只跑少量 epoch，验证流程是否正常
    """
    cfg = Config()

    # ==== 可调的「快速训练」参数 ====
    SAMPLE_SIZE = 1_000 # 从训练集中随机采样这么多条交互（可根据机器性能调整）
    QUICK_EPOCHS = 1      # 快速训练只跑这么多轮

    # 1. 加载图结构
    print("[quick_train] 加载图结构 ...")
    edge_list, num_users, num_items = load_graph()
    print(f"[quick_train] 图加载完成: 边数 = {edge_list.shape[0]}, "
          f"用户数 = {num_users}, 物品数 = {num_items}")

    # 2. 构建邻居采样器
    print(f"[quick_train] 构建 NeighborSampler, num_neighbors = {cfg.num_neighbors}")
    sampler = NeighborSampler(edge_list, cfg.num_neighbors)

    # 3. 构建完整训练数据集，然后随机选一部分做子集
    print("[quick_train] 加载训练数据集 data/processed/train.parquet ...")
    full_dataset = InteractionDataset("data/processed/train.parquet", num_items)
    full_len = len(full_dataset)
    print(f"[quick_train] 完整训练样本数 = {full_len}")

    sample_size = min(SAMPLE_SIZE, full_len)
    indices = np.random.choice(full_len, size=sample_size, replace=False)
    subset = torch.utils.data.Subset(full_dataset, indices)
    print(f"[quick_train] 实际使用的子集样本数 = {len(subset)}")

    # 4. 自定义 collate（与 main_baseline 一致）
    def _collate(batch):
        processed = []
        for u, pos, neg in batch:
            nodes, node2idx = sampler.sample([u, pos, neg])
            processed.append(((nodes, node2idx), (u, pos, neg)))
        return collate_subgraphs(processed)

    # 5. DataLoader
    print(f"[quick_train] 构建 DataLoader: batch_size = {cfg.batch_size}, num_workers = 0")
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
    )

    # 6. 设备与模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[quick_train] 使用设备: {device}")

    print("[quick_train] 构建 GraphSAGERecommender 模型 ...")
    model = GraphSAGERecommender(
        num_users,
        num_items,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    edge_index = torch.tensor(edge_list.T, dtype=torch.long).to(device)

    # 7. 训练（只跑 QUICK_EPOCHS 轮）
    print(f"[quick_train] 开始快速训练: epochs = {QUICK_EPOCHS}")
    train_baseline(
        model,
        loader,
        edge_index,
        optimizer,
        device=device,
        epochs=QUICK_EPOCHS,
    )
    print("[quick_train] 快速训练结束。")


if __name__ == "__main__":
    main()


