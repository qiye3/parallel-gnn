# core/sampler/collate_fn.py

from typing import List, Dict, Tuple


def collate_subgraphs(batch):
    """
    支持两种格式：
        1. ((sub_nodes, node2idx), (u,pos,neg))
        2. ((sub_nodes, node2idx, edges_sub), (u,pos,neg))
    """

    nodes_list: List[List[int]] = []
    maps_list: List[Dict[int, int]] = []
    edges_list: List = []
    triples: List[Tuple[int, int, int]] = []

    for item, triple in batch:
        triples.append(triple)

        # 兼容两种返回格式
        if len(item) == 2:
            sub_nodes, node2idx = item
            edges_sub = None
        elif len(item) == 3:
            sub_nodes, node2idx, edges_sub = item
        else:
            raise ValueError("collate_subgraphs: item 长度不正确")

        nodes_list.append(sub_nodes)
        maps_list.append(node2idx)
        edges_list.append(edges_sub)

    return nodes_list, maps_list, triples
