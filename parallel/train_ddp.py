# parallel/train_ddp.py
# ----------------------
# 基于 torch.distributed 的 DDP 训练入口。
# 通过 torchrun 启动多个进程，每个进程绑定一个 GPU，进行数据并行训练。

import os
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from core.loader.datasets import InteractionDataset
from core.sampler.collate_fn import collate_subgraphs
from core.sampler.neighbor_sampler import NeighborSampler
from core.models.graphsage import GraphSAGERecommender
from core.train.train_baseline import train_baseline
from core.config import Config


def setup(rank: int, world: int):
    """
    初始化 DDP 通信环境。

    参数:
        rank  : 当前进程在进程组中的序号
        world : 总进程数（world size）
    """
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method="env://",  # 默认使用环境变量进行初始化
        rank=rank,
        world_size=world,
    )
    if torch.cuda.is_available():
        # 每个进程绑定一个 GPU，约定 rank 与 GPU id 一致
        torch.cuda.set_device(rank)


def main_ddp():
    """
    DDP 训练主函数，由 torchrun -m parallel.train_ddp 调用。

    环境变量约定：
        - RANK       : 当前进程编号
        - WORLD_SIZE : 总进程数
    """
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    setup(rank, world)

    cfg = Config()

    # 每个进程各自从磁盘加载图数据（也可以优化为只在 rank0 读然后广播）
    edge_list = np.load("data/graph/edge_list.npy")
    num_users, num_items = np.load("data/graph/num_info.npy")

    # 构建 Dataset 与邻居采样器（目前在每个进程中独立构建）
    dataset = InteractionDataset("data/processed/train.parquet", num_items)
    sampler = NeighborSampler(edge_list, cfg.num_neighbors)

    # 分布式采样器：负责将数据划分给不同 rank
    dist_sampler = DistributedSampler(dataset, num_replicas=world, rank=rank)

    def _collate(batch):
        """
        与 baseline 相同的子图采样逻辑，只是此处的 batch 已由 DistributedSampler 划分。
        """
        processed = []
        for u, pos, neg in batch:
            nodes, node2idx = sampler.sample([u, pos, neg])
            processed.append(((nodes, node2idx), (u, pos, neg)))
        return collate_subgraphs(processed)

    # 每个进程本地的 DataLoader，只会看到自己负责的那一部分数据
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=dist_sampler,
        num_workers=2,
        collate_fn=_collate,
    )

    # 不同进程绑定到不同设备
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    # 构建模型，并移动到本进程对应的设备上
    model = GraphSAGERecommender(
        num_users,
        num_items,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    ).to(device)

    # 包装为 DDP 模型，实现梯度同步
    ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=cfg.lr)

    edge_index = torch.tensor(edge_list.T, dtype=torch.long).to(device)

    # 直接复用基线训练逻辑，此时 model 已经是 DDP 封装
    train_baseline(ddp_model, loader, edge_index, optimizer, device=device, epochs=cfg.epochs)

    # 训练结束后销毁进程组
    dist.destroy_process_group()
