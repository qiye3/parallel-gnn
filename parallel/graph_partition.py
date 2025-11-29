# parallel/graph_partition.py

import numpy as np


def partition_nodes(num_nodes, num_parts):
    part_size = num_nodes // num_parts
    parts = []
    for i in range(num_parts):
        start = i * part_size
        end = (i + 1) * part_size if i != num_parts - 1 else num_nodes
        parts.append((start, end))
    return parts


def build_subgraph(edge_list, node_range):
    s, e = node_range
    mask = (edge_list[:, 0] >= s) & (edge_list[:, 0] < e)
    return edge_list[mask].copy()
