"""
scripts/quick_train.py
----------------------
一个用于「快速训练 / 烟雾测试」的小脚本：
    - 支持单进程或多进程 DataLoader
    - 可选的随机采样训练集的一小部分样本
    - 只跑少量 epoch
    - 方便检查代码是否能正常跑通、观察日志输出

使用方式（在项目根目录下）：
    python scripts/quick_train.py                    # 单进程
    python scripts/quick_train.py --mp --workers 4   # 多进程
    python scripts/quick_train.py --sample 1000      # 采样1000个样本
"""

import os
import sys
import argparse
import numpy as np
import torch

# ---------- 确保可以导入项目根目录下的 core 包 ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config import Config
from core.loader.datasets import InteractionDataset
from core.train.setup import setup_training
from core.train.train_baseline import train_baseline


def main():
    """
    快速训练主函数：
        - 支持单进程或多进程 DataLoader
        - 可选的随机采样训练集的一小部分样本
        - 只跑少量 epoch，验证流程是否正常
    """
    parser = argparse.ArgumentParser(description="快速训练脚本")
    parser.add_argument("--mp", action="store_true", help="使用多进程 DataLoader")
    parser.add_argument("--workers", type=int, default=4, help="worker 数量（仅当 --mp 时有效）")
    parser.add_argument("--sample", type=int, default=None, help="随机采样样本数（None 表示使用全部数据）")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    args = parser.parse_args()
    
    cfg = Config()
    prefix = "[quick_train_mp]" if args.mp else "[quick_train]"
    
    # 加载图结构（用于获取 num_items）
    from core.loader.utils_io import load_graph
    _, _, num_items = load_graph()
    
    # 如果指定了采样，需要先加载数据集创建子集
    if args.sample is not None:
        print(f"{prefix} 创建数据子集（采样 {args.sample} 个样本）...")
        full_dataset = InteractionDataset("data/processed/train.parquet", num_items)
        full_len = len(full_dataset)
        sample_size = min(args.sample, full_len)
        indices = np.random.choice(full_len, size=sample_size, replace=False)
        subset = torch.utils.data.Subset(full_dataset, indices)
        
        # 使用 setup_training 但需要手动替换数据集
        # 先正常初始化
        model, _, edge_index, optimizer, device, num_users, num_items, edge_list = setup_training(
            cfg, use_mp=args.mp, num_workers=args.workers
        )
        
        # 然后为子集创建新的 DataLoader
        from core.sampler.neighbor_sampler import NeighborSampler
        from core.sampler.collate_fn import collate_subgraphs
        sampler = NeighborSampler(edge_list, cfg.num_neighbors)
        
        def _collate(batch):
            processed = []
            for u, pos, neg in batch:
                nodes, node2idx = sampler.sample([u, pos, neg])
                processed.append(((nodes, node2idx), (u, pos, neg)))
            return collate_subgraphs(processed)
        
        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=args.workers if args.mp else 0,
            collate_fn=_collate,
        )
        print(f"{prefix} 实际使用的子集样本数 = {len(subset)}")
    else:
        # 使用完整数据集
        print(f"{prefix} 初始化训练环境...")
        model, loader, edge_index, optimizer, device, num_users, num_items, edge_list = setup_training(
            cfg, use_mp=args.mp, num_workers=args.workers
        )
    
    print(f"{prefix} 使用设备: {device}")
    print(f"{prefix} 开始快速训练: epochs = {args.epochs}")
    
    train_baseline(
        model,
        loader,
        edge_index,
        optimizer,
        device=device,
        epochs=args.epochs,
    )
    print(f"{prefix} 快速训练结束。")


if __name__ == "__main__":
    main()


