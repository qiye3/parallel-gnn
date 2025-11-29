# parallel/dataloader_mp.py
# -------------------------
# 多进程版本的数据加载器构造函数。
# 主要思想：
#   - 使用 PyTorch DataLoader 的多进程机制（num_workers > 0）
#   - 在每个 worker 内部维护一个独立的 NeighborSampler 实例
#   - 采样逻辑在 worker 进程中并行执行，从而提升数据准备速度

import torch
from torch.utils.data import DataLoader
from core.loader.datasets import InteractionDataset
from core.sampler.collate_fn import collate_subgraphs
from core.sampler.neighbor_sampler import NeighborSampler


def create_mp_dataloader(
    parquet_path: str,
    num_items: int,
    edge_list,
    num_neighbors,
    batch_size: int = 1024,
    num_workers: int = 4,
):
    """
    创建带多进程子图采样的 DataLoader。

    参数:
        parquet_path : 训练数据 parquet 路径
        num_items    : 物品总数（用于负采样）
        edge_list    : 图的边列表（numpy 数组或等价形式），交给 NeighborSampler 使用
        num_neighbors: 每层采样的邻居数列表，如 [15, 10]
        batch_size   : DataLoader 的 batch 大小
        num_workers  : DataLoader 使用的 worker 数量（大于 0 即启用多进程）
    """
    # 基础交互数据集
    dataset = InteractionDataset(parquet_path, num_items)

    def worker_init(worker_id: int):
        """
        每个 worker 进程初始化时调用。

        在此为当前 worker 挂载一个独立的 NeighborSampler，
        避免不同进程之间共享复杂对象带来的潜在问题。
        """
        worker_info = torch.utils.data.get_worker_info()
        # 在 worker_info 上动态添加 sampler 属性，供 _collate 使用
        worker_info.sampler = NeighborSampler(edge_list, num_neighbors=num_neighbors)

    def _collate(batch):
        """
        自定义 collate 函数：在 worker 进程中完成子图采样。

        输入:
            batch: 由 Dataset.__getitem__ 返回的若干 (u, pos, neg) 三元组
        """
        # 拿到当前 worker 对应的 NeighborSampler
        sampler = torch.utils.data.get_worker_info().sampler
        processed = []
        for (u, pos, neg) in batch:
            # 对每个样本基于 (u, pos, neg) 三个种子节点采样子图
            nodes, node2idx = sampler.sample([u, pos, neg])
            processed.append(((nodes, node2idx), (u, pos, neg)))
        # 再统一交给 collate_subgraphs 合并为 batch 结构
        return collate_subgraphs(processed)

    # 构建 DataLoader，开启多进程与持久化 worker
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        worker_init_fn=worker_init,
        persistent_workers=True,
    )
