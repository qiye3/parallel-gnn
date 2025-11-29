# core/loader/utils_io.py
# ------------------------
# 图结构与相关元信息的 IO 工具函数。

import numpy as np


def load_graph():
    """
    从磁盘加载训练所需的图结构信息。

    约定的文件路径：
        - data/graph/edge_list.npy  : 形状为 [E, 2] 的 numpy 数组，每一行是一条 (user, item) 边
        - data/graph/num_info.npy   : 长度为 2 的数组 [num_users, num_items]

    返回值:
        edge_list (np.ndarray): 边列表，形状 [E, 2]，元素类型为 int
        num_users (int): 用户节点数量
        num_items (int): 物品节点数量
    """
    # 读取边列表，包含所有训练边 (uid, iid)
    edge_list = np.load("data/graph/edge_list.npy")
    # 读取用户数与物品数的统计信息
    num_users, num_items = np.load("data/graph/num_info.npy")
    return edge_list, int(num_users), int(num_items)
