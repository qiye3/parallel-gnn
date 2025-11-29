# core/loader/build_graph.py
# ---------------------------------------
# 适配 MovieLens-10M (:: 分隔符) 的构图脚本

import os
import pandas as pd
import numpy as np

RATINGS_PATH = "data/raw/ratings.dat"   # ML-10M 文件


def load_ml10m_chunked(path=RATINGS_PATH, chunksize=500_000):
    """
    按块读取 MovieLens-10M 的 ratings.dat
    数据格式: UserID::MovieID::Rating::Timestamp
    """
    return pd.read_csv(
        path,
        sep="::",
        engine="python",        # "::" 需要 python 引擎
        names=["userId", "movieId", "rating", "timestamp"],
        dtype={
            "userId": np.int32,
            "movieId": np.int32,
            "rating": np.float32,
            "timestamp": np.int64,
        },
        chunksize=chunksize,
    )


def remap_ids_full():
    """
    Pass 1：构建 userId / movieId 的连续 ID 映射
    """
    print("=== Pass 1: scanning all IDs (ML-10M) ===")
    user_set = set()
    item_set = set()

    for chunk in load_ml10m_chunked():
        user_set.update(chunk["userId"].unique())
        item_set.update(chunk["movieId"].unique())

    user_list = sorted(list(user_set))
    item_list = sorted(list(item_set))

    num_users = len(user_list)
    num_items = len(item_list)

    print(f"  Users: {num_users}")
    print(f"  Items: {num_items}")

    uid_map = {u: i for i, u in enumerate(user_list)}
    iid_map = {m: i + num_users for i, m in enumerate(item_list)}

    return uid_map, iid_map, num_users, num_items


def temporal_split(df, val_ratio=0.1, test_ratio=0.1):
    """
    按时间戳排序，切分 train/val/test
    """
    df = df.sort_values("timestamp")
    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test = df.iloc[-n_test:]
    val = df.iloc[-(n_test + n_val): -n_test]
    train = df.iloc[: -(n_test + n_val)]
    return train, val, test


def build_user_item_graph():
    """
    适配 ML-10M 的构图脚本。
    输出：
        data/processed/train.parquet
        data/processed/val.parquet
        data/processed/test.parquet
        data/graph/edge_list.npy
        data/graph/num_info.npy
    """
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/graph", exist_ok=True)

    print("=== Building MovieLens-10M graph ===")

    # Pass 1：扫描所有 userId / movieId
    uid_map, iid_map, num_users, num_items = remap_ids_full()

    # Pass 2：重新读取 ratings.dat 并重新编码
    print("=== Pass 2: remapping IDs & concatenating ===")
    dfs = []

    for chunk in load_ml10m_chunked():
        df = pd.DataFrame({
            "uid": chunk["userId"].map(uid_map).astype(np.int32),
            "iid": chunk["movieId"].map(iid_map).astype(np.int32),
            "rating": chunk["rating"].astype(np.float32),
            "timestamp": chunk["timestamp"].astype(np.int64),
        })
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    del dfs

    print(f"  Total interactions = {len(df_all)}")

    print("=== Splitting train/val/test ===")
    train, val, test = temporal_split(df_all)

    train.to_parquet("data/processed/train.parquet")
    val.to_parquet("data/processed/val.parquet")
    test.to_parquet("data/processed/test.parquet")

    print("=== Building edge_list from TRAIN only ===")
    edge_list = np.vstack([train["uid"].values, train["iid"].values]).T
    np.save("data/graph/edge_list.npy", edge_list.astype(np.int32))

    np.save("data/graph/num_info.npy", np.array([num_users, num_items], dtype=np.int64))

    print("=== Done ===")
    print(f" Users = {num_users}")
    print(f" Items = {num_items}")
    print(f" Train edges = {edge_list.shape[0]}")
