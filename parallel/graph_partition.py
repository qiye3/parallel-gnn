# parallel/graph_partition.py
# ---------------------------
# 图划分相关的简单工具函数，用于将节点 ID 空间按区间切分，并构造对应子图。

import numpy as np


def partition_nodes(num_nodes: int, num_parts: int):
    """
    将 [0, num_nodes) 的节点 ID 区间平均划分为 num_parts 段。

    返回:
        parts: List[Tuple[int, int]]，每个元素为一个闭开区间 [start, end)
    """
    part_size = num_nodes // num_parts
    parts = []
    for i in range(num_parts):
        start = i * part_size
        # 最后一段包含余数
        end = (i + 1) * part_size if i != num_parts - 1 else num_nodes
        parts.append((start, end))
    return parts


def build_subgraph(edge_list: np.ndarray, node_range):
    """
    根据给定的节点区间，构建只包含该区间内节点的子图边列表。

    参数:
        edge_list : [E, 2] 的边列表数组
        node_range: (s, e) 形式的元组，表示节点区间 [s, e)

    返回:
        sub_edge_list: 仅保留满足 s <= src < e 的边
    """
    s, e = node_range
    # 只筛选出源节点位于该区间内的边
    mask = (edge_list[:, 0] >= s) & (edge_list[:, 0] < e)
    return edge_list[mask].copy()
