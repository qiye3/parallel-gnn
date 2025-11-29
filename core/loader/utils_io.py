# core/loader/utils_io.py

import numpy as np


def load_graph():
    """
    加载 edge_list.npy 和 num_info.npy
    返回:
        edge_list: np.ndarray [[u, i], ...]
        num_users: int
        num_items: int
    """
    edge_list = np.load("data/graph/edge_list.npy")
    num_users, num_items = np.load("data/graph/num_info.npy")
    return edge_list, int(num_users), int(num_items)
