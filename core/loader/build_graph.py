# core/loader/build_graph.py
# ---------------------------
# 负责从原始 MovieLens-32M 评分数据构建：
#   1. 连续、紧凑的用户 / 物品 ID 映射（从 0 开始）
#   2. 训练 / 验证 / 测试集划分（按时间戳进行时序划分）
#   3. 仅基于训练集构造 user-item 二部图的边列表 edge_list.npy
#   4. 保存用户数 / 物品数统计信息 num_info.npy

import os
import pandas as pd
import numpy as np

RATINGS_PATH = "data/raw/ratings.csv"


def load_movielens_32m_chunked(path: str = RATINGS_PATH, chunksize: int = 2_000_000):
    """
    分块读取 MovieLens-32M ratings.csv，避免一次性将全部 3200 万条记录读入内存。

    参数:
        path      : 原始 ratings.csv 的路径
        chunksize : 每个 chunk 的行数（即一次读取多少条记录）

    返回:
        一个 Pandas 的 TextFileReader，可迭代得到 DataFrame chunk
    """
    return pd.read_csv(
        path,
        chunksize=chunksize,
        dtype={
            "userId": np.int32,
            "movieId": np.int32,
            "rating": np.float32,
            "timestamp": np.int64,
        },
    )


def remap_ids_full():
    """
    两遍扫描原始数据，构建用户 / 物品 ID 的紧凑映射。

    流程：
        1. 第一遍遍历所有 chunk，收集 userId / movieId 的去重集合
        2. 将集合排序后，构建从原始 ID -> 连续 ID 的映射字典

    返回:
        uid_map (dict) : 原始 userId -> 新的 [0, num_users) 索引
        iid_map (dict) : 原始 movieId -> 新的 [num_users, num_users+num_items) 索引
        num_users (int): 用户总数
        num_items (int): 物品总数
    """
    print("=== Pass 1: scanning all IDs ===")
    user_set = set()
    item_set = set()

    # 遍历所有 chunk，收集 userId 与 movieId
    for chunk in load_movielens_32m_chunked():
        user_set.update(chunk["userId"].unique())
        item_set.update(chunk["movieId"].unique())

    # 将 set 转为有序列表，保证映射的可复现性
    user_list = sorted(list(user_set))
    item_list = sorted(list(item_set))

    num_users = len(user_list)
    num_items = len(item_list)

    print(f"  Users: {num_users}")
    print(f"  Items: {num_items}")

    # 用户 ID 从 0 开始，物品 ID 从 num_users 开始，构成一个「整体图」上的索引空间
    uid_map = {u: i for i, u in enumerate(user_list)}
    iid_map = {m: i + num_users for i, m in enumerate(item_list)}

    return uid_map, iid_map, num_users, num_items


def temporal_split(df: pd.DataFrame, val_ratio: float = 0.1, test_ratio: float = 0.1):
    """
    按时间戳进行时序划分，将完整 DataFrame 划分为 train/val/test。

    策略：
        - 先整体按 timestamp 升序排序
        - 最后 test_ratio 部分作为测试集
        - 倒数第 (val_ratio + test_ratio) 部分中的前 val_ratio 部分作为验证集
        - 剩余前面的部分作为训练集
    """
    df = df.sort_values("timestamp")
    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test = df.iloc[-n_test:]
    val = df.iloc[-(n_test + n_val) : -n_test]
    train = df.iloc[: -(n_test + n_val)]
    return train, val, test


def build_user_item_graph():
    """
    构建用户-物品二部图及其对应的训练 / 验证 / 测试集。

    生成文件：
        - data/processed/train.parquet
        - data/processed/val.parquet
        - data/processed/test.parquet
        - data/graph/edge_list.npy  (仅使用训练集边)
        - data/graph/num_info.npy   ([num_users, num_items])
    """
    # 确保输出目录存在
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/graph", exist_ok=True)

    print("=== Building MovieLens-32M graph ===")

    # -------- Pass 1: 扫描所有 ID，构建 ID 映射 -------- #
    uid_map, iid_map, num_users, num_items = remap_ids_full()

    # -------- Pass 2: 基于映射重新编码并拼接所有 chunk -------- #
    print("=== Pass 2: remapping IDs & concatenating ===")
    dfs = []

    for chunk in load_movielens_32m_chunked():
        # 将原始 userId / movieId 映射到新的 uid / iid
        df = pd.DataFrame(
            {
                "uid": chunk["userId"].map(uid_map).astype(np.int32),
                "iid": chunk["movieId"].map(iid_map).astype(np.int32),
                "rating": chunk["rating"].astype(np.float32),
                "timestamp": chunk["timestamp"].astype(np.int64),
            }
        )
        dfs.append(df)

    # 按行拼接所有 chunk，得到完整交互数据
    df_all = pd.concat(dfs, ignore_index=True)
    del dfs

    print("  Total interactions:", len(df_all))

    # -------- Temporal Split：划分 train / val / test -------- #
    print("=== Splitting train/val/test ===")
    train, val, test = temporal_split(df_all)

    # 将划分结果保存为 parquet，供训练 / 评估使用
    train.to_parquet("data/processed/train.parquet")
    val.to_parquet("data/processed/val.parquet")
    test.to_parquet("data/processed/test.parquet")

    # -------- 基于训练集构建边列表 edge_list -------- #
    print("=== Building edge list ===")
    # 只使用训练集中的交互构建图，避免「泄漏」验证和测试信息
    edge_list = np.vstack([train["uid"].values, train["iid"].values]).T
    np.save("data/graph/edge_list.npy", edge_list.astype(np.int32))

    # 保存用户数与物品数
    np.save("data/graph/num_info.npy", np.array([num_users, num_items], dtype=np.int64))

    print("=== Done ===")
    print(f" Users: {num_users}")
    print(f" Items: {num_items}")
    print(f" Train edges: {edge_list.shape[0]}")
