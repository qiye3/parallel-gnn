# core/sampler/collate_fn.py

from typing import List, Tuple, Dict


def collate_subgraphs(batch):
    """
    batch: list of ((sub_nodes, node2idx), (u, pos, neg))

    返回:
        nodes_list: List[List[int]]
        maps_list: List[Dict[int,int]]
        triples: List[(u, pos, neg)]
    """
    nodes_list = []
    maps_list = []
    triples = []

    for (sub_nodes, node2idx), (u, pos, neg) in batch:
        nodes_list.append(sub_nodes)
        maps_list.append(node2idx)
        triples.append((u, pos, neg))

    return nodes_list, maps_list, triples
