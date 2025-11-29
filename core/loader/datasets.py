# core/loader/datasets.py
# ------------------------
# 数据集定义：用于从 parquet 文件中读取 (user, item) 交互，
# 并在 __getitem__ 中完成简单的负采样，供训练使用。

import numpy as np
import pandas as pd
import torch


class InteractionDataset(torch.utils.data.Dataset):
    """
    用户-物品交互数据集。

    每个样本返回一个三元组 (u, pos, neg)：
        - u   : 用户 ID（内部重新映射后的整型索引）
        - pos : 正样本物品 ID（用户真实交互过的物品）
        - neg : 负样本物品 ID（从所有物品中随机采样）
    """

    def __init__(self, parquet_path: str, num_items: int):
        """
        参数:
            parquet_path: 训练 / 验证 / 测试 parquet 文件路径
            num_items   : 物品总数，用于负采样的上界
        """
        # 从 parquet 读取所有交互，假定已通过 build_graph 预处理为 uid / iid 列
        df = pd.read_parquet(parquet_path)
        # 用户 ID 序列（numpy 数组，节省内存）
        self.users = df["uid"].values
        # 物品 ID 序列
        self.items = df["iid"].values
        # 物品数量（负采样时用作 randint 上界）
        self.num_items = int(num_items)

    def __len__(self) -> int:
        """返回交互样本数量。"""
        return len(self.users)

    def __getitem__(self, idx: int):
        """
        根据索引返回一个训练样本三元组 (u, pos, neg)。

        负采样策略：
            - 从 [0, num_items) 区间内均匀随机采样一个物品作为负样本
            - 简单但高效，适合作为 baseline
        """
        # 取出当前样本对应的用户 ID 和正样本物品 ID
        u = int(self.users[idx])
        pos = int(self.items[idx])
        # 从所有物品中随机采样一个负样本（不排除与正样本相同的情况，简化实现）
        neg = np.random.randint(0, self.num_items)
        return u, pos, neg
