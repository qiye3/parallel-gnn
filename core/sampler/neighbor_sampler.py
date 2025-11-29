# core/sampler/neighbor_sampler.py

import numpy as np
from typing import List, Dict, Tuple


class NeighborSampler:
    """
    串行邻居采样，用于 GraphSAGE-style 扩散。
    输入:
        edge_list: [E, 2]
        num_neighbors: 如 [15, 10]
    """

    def __init__(self, edge_list: np.ndarray, num_neighbors: List[int]):
        self.edge_list = edge_list
        self.num_neighbors = num_neighbors
        self.num_nodes = int(edge_list.max() + 1)

        # 构建无向图邻接表
        self.adj = [[] for _ in range(self.num_nodes)]
        for u, v in edge_list:
            self.adj[u].append(v)
            self.adj[v].append(u)

    def sample(self, seeds: List[int]) -> Tuple[List[int], Dict[int, int]]:
        """
        给定 seeds，进行多层采样。
        返回:
            all_nodes: 子图节点列表
            node2idx: 原 ID 到子图 index 的映射
        """
        layers = [list(seeds)]
        frontier = list(seeds)

        for k in self.num_neighbors:
            nf = []
            for nid in frontier:
                neigh = self.adj[nid]
                if len(neigh) <= k:
                    sampled = neigh
                else:
                    sampled = list(np.random.choice(neigh, k, replace=False))
                nf.extend(sampled)
            frontier = list(set(nf))
            layers.append(frontier)

        all_nodes = sorted(set().union(*layers))
        node2idx = {nid: i for i, nid in enumerate(all_nodes)}

        return all_nodes, node2idx
