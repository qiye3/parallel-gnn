# core/sampler/neighbor_sampler.py

import numpy as np
from typing import List, Dict, Tuple


class NeighborSampler:
    """
    串行邻居采样器：
      - sample(): 返回多层采样的节点集合（给 baseline 用）
      - sample_subgraph(): 返回节点集合 + 子图边（给 mini-batch GNN 训练用）
    """

    def __init__(self, edge_list: np.ndarray, num_neighbors: List[int]):
        """
        edge_list: [E, 2]，每行 (u, v)
        num_neighbors: 每一层采样的邻居个数，例如 [15, 10]
        """
        self.edge_list = edge_list
        self.num_neighbors = num_neighbors
        self.num_nodes = int(edge_list.max() + 1)

        # 构建无向邻接表
        self.adj = [[] for _ in range(self.num_nodes)]
        for u, v in edge_list:
            self.adj[u].append(v)
            self.adj[v].append(u)

    # -------- 给 baseline / quick_train 用的老接口 --------
    def sample(self, seeds: List[int]) -> Tuple[List[int], Dict[int, int]]:
        layers = [list(seeds)]
        frontier = list(seeds)

        for k in self.num_neighbors:
            nf = []
            for nid in frontier:
                neigh = self.adj[nid]
                if len(neigh) == 0:
                    continue
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

    # -------- 给 mini-batch GNN 训练用的新接口 --------
    def sample_subgraph(
        self, seeds: List[int]
    ) -> Tuple[List[int], Dict[int, int], np.ndarray]:
        """
        从种子节点出发，按 num_neighbors 扩散，返回：
          - sub_nodes: 子图中所有节点的“全局 id”列表
          - node2idx : 全局id -> 子图局部索引 [0..N_sub-1]
          - edge_index_sub: [E_sub, 2]，使用局部索引的子图边
        """
        visited = set(seeds)
        frontier = set(seeds)

        # 多层邻居扩散
        for k in self.num_neighbors:
            next_frontier = set()
            for nid in frontier:
                neigh = self.adj[nid]
                if len(neigh) == 0:
                    continue
                if len(neigh) <= k:
                    sampled = neigh
                else:
                    sampled = np.random.choice(neigh, k, replace=False)
                for v in sampled:
                    if v not in visited:
                        visited.add(v)
                        next_frontier.add(v)
            frontier = next_frontier

        sub_nodes = sorted(visited)
        node2idx = {nid: i for i, nid in enumerate(sub_nodes)}

        # 在子图节点集合内构建边（局部索引）
        edges = []
        for u in sub_nodes:
            for v in self.adj[u]:
                if v in node2idx:
                    edges.append((node2idx[u], node2idx[v]))

        if len(edges) == 0:
            edge_index_sub = np.zeros((0, 2), dtype=np.int64)
        else:
            edge_index_sub = np.array(edges, dtype=np.int64)

        return sub_nodes, node2idx, edge_index_sub
