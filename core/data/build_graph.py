# core/data/build_graph.py

import os
import pandas as pd
import numpy as np

RATINGS_PATH = "data/raw/ratings.csv"


def load_movielens_32m_chunked(path=RATINGS_PATH, chunksize=2_000_000):
    """
    分块读取 MovieLens-32M ratings.csv
    每次读 chunksize 条，避免一次性占用大量内存。
    """
    return pd.read_csv(
        path,
        chunksize=chunksize,
        dtype={
            "userId": np.int32,
            "movieId": np.int32,
            "rating": np.float32,
            "timestamp": np.int64,
        }
    )


def remap_ids_full():
    """
    第一遍扫描：收集所有 userId 和 movieId
    第二遍扫描：生成最终 train/val/test
    """
    print("=== Pass 1: scanning all IDs ===")
    user_set = set()
    item_set = set()

    for chunk in load_movielens_32m_chunked():
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
    df = df.sort_values("timestamp")
    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test = df.iloc[-n_test:]
    val = df.iloc[-(n_test + n_val):-n_test]
    train = df.iloc[:-(n_test + n_val)]
    return train, val, test


def build_user_item_graph():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/graph", exist_ok=True)

    print("=== Building MovieLens-32M graph ===")

    # -------- Pass 1: Scan for all IDs (32M rows) -------- #
    uid_map, iid_map, num_users, num_items = remap_ids_full()

    # -------- Pass 2: Remap and concatenate -------- #
    print("=== Pass 2: remapping IDs & concatenating ===")
    dfs = []

    for chunk in load_movielens_32m_chunked():
        df = pd.DataFrame({
            "uid": chunk["userId"].map(uid_map).astype(np.int32),
            "iid": chunk["movieId"].map(iid_map).astype(np.int32),
            "rating": chunk["rating"].astype(np.float32),
            "ts": chunk["timestamp"].astype(np.int64),
        })
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    del dfs

    print("  Total interactions:", len(df_all))

    # -------- Temporal Split -------- #
    print("=== Splitting train/val/test ===")
    train, val, test = temporal_split(df_all)

    train.to_parquet("data/processed/train.parquet")
    val.to_parquet("data/processed/val.parquet")
    test.to_parquet("data/processed/test.parquet")

    # -------- Build Edge List (using train only) -------- #
    print("=== Building edge list ===")
    edge_list = np.vstack([train["uid"].values, train["iid"].values]).T
    np.save("data/graph/edge_list.npy", edge_list.astype(np.int32))
    np.save("data/graph/num_info.npy",
            np.array([num_users, num_items], dtype=np.int64))

    print("=== Done ===")
    print(f" Users: {num_users}")
    print(f" Items: {num_items}")
    print(f" Train edges: {edge_list.shape[0]}")
