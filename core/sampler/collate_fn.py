# core/sampler/collate_fn.py
# ---------------------------
# DataLoader 的自定义 collate 函数：将若干子图样本打包成一个 batch。

from typing import List, Tuple, Dict


def collate_subgraphs(batch):
    """
    将若干条样本合并成 batch。

    输入 batch 结构:
        batch: list[ ((sub_nodes, node2idx), (u, pos, neg)) ]

        其中：
            - sub_nodes: List[int]，当前样本子图中的所有节点 ID
            - node2idx : Dict[int, int]，原图节点 ID -> 子图节点下标
            - (u, pos, neg): 当前样本的用户 / 正样本物品 / 负样本物品 ID

    返回:
        nodes_list: List[List[int]]
            - 每个元素是一个样本对应的子图节点列表
        maps_list: List[Dict[int, int]]
            - 每个元素是该样本的 node2idx 映射
        triples: List[Tuple[int, int, int]]
            - 每个元素为 (u, pos, neg) 三元组
    """
    nodes_list: List[List[int]] = []
    maps_list: List[Dict[int, int]] = []
    triples: List[Tuple[int, int, int]] = []

    # 逐个样本拆包
    for (sub_nodes, node2idx), (u, pos, neg) in batch:
        nodes_list.append(sub_nodes)
        maps_list.append(node2idx)
        triples.append((u, pos, neg))

    return nodes_list, maps_list, triples
