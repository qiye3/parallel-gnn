# core/sampler/neighbor_sampler.py
# --------------------------------
# GraphSAGE 风格的多层邻居采样器，用于从大图中抽取局部子图。

import numpy as np
from typing import List, Dict, Tuple


class NeighborSampler:
    """
    串行邻居采样器（CPU 版本）。

    用途：
        - 在大规模图上，针对一小批「种子节点」(seeds)，
          逐层采样其 k-hop 邻居，构建用于 GNN 训练的子图。

    参数:
        edge_list   : 形状为 [E, 2] 的 numpy 数组，每行 (u, v) 为一条无向边
        num_neighbors: 每一层采样的邻居数列表，如 [15, 10] 表示两层采样
    """

    def __init__(self, edge_list: np.ndarray, num_neighbors: List[int]):
        self.edge_list = edge_list
        self.num_neighbors = num_neighbors
        # 假定节点 ID 从 0 开始连续编号，最大 ID + 1 即为节点数
        self.num_nodes = int(edge_list.max() + 1)

        # 构建无向图邻接表：adj[i] 为与节点 i 相连的所有邻居节点列表
        self.adj = [[] for _ in range(self.num_nodes)]
        for u, v in edge_list:
            self.adj[u].append(v)
            self.adj[v].append(u)

    def sample(self, seeds: List[int]) -> Tuple[List[int], Dict[int, int]]:
        """
        给定 seeds，进行多层邻居采样，返回子图节点列表及其映射。

        采样流程：
            - 初始化第 0 层为 seeds
            - 对于每一层 k，在当前 frontier 的每个节点上随机采 k 个邻居
            - 将新采样到的所有邻居集合作为下一层 frontier
            - 最后将所有层的节点取并集，形成子图节点集

        返回:
            all_nodes: 子图中所有出现过的节点 ID（升序排列）
            node2idx : 将原图节点 ID 映射到子图局部索引的字典
        """
        # layers[i] 记录第 i 层的节点列表，layers[0] 即 seeds
        layers = [list(seeds)]
        frontier = list(seeds)

        # 逐层扩散采样
        for k in self.num_neighbors:
            nf = []
            for nid in frontier:
                neigh = self.adj[nid]
                if len(neigh) <= k:
                    # 邻居不足时，全部保留
                    sampled = neigh
                else:
                    # 从邻居中不放回随机采样 k 个
                    sampled = list(np.random.choice(neigh, k, replace=False))
                nf.extend(sampled)
            # 去重后的新 frontier
            frontier = list(set(nf))
            layers.append(frontier)

        # 将所有层的节点取并集，并排序，得到子图节点列表
        all_nodes = sorted(set().union(*layers))
        # 构建原始 ID -> 子图局部索引的映射
        node2idx = {nid: i for i, nid in enumerate(all_nodes)}

        return all_nodes, node2idx
