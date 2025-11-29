# parallel/dist_inference.py
# --------------------------
# 分布式推理：在多进程环境下对整图做一次前向传播，并将嵌入结果在所有进程之间聚合。

import torch
import torch.distributed as dist


def distributed_inference(model, edge_index):
    """
    在 DDP 环境下对整图进行一次前向传播，并在进程间汇总节点嵌入。

    注意：
        这里的实现比较简单 —— 每个进程都调用 forward_full 计算整图 embedding，
        然后仅保留自己负责的那一部分区间，再使用 all_gather 聚合。
        在实际大规模场景中，可以改造成真正的「分块前向」以节省显存。
    """
    rank = dist.get_rank()
    world = dist.get_world_size()

    # 每个 worker 负责一段连续的节点区间 [start, end)
    num_nodes = model.num_nodes
    part = num_nodes // world
    start = rank * part
    end = (rank + 1) * part if rank < world - 1 else num_nodes

    with torch.no_grad():
        # 当前实现中，所有进程都计算 full_emb；
        # 也可以优化为仅由部分进程计算，再通过通信交换。
        full_emb = model.forward_full(edge_index)
        local_emb = full_emb[start:end]

    # gather 所有进程负责区间的 embedding
    emb_list = [torch.zeros_like(local_emb) for _ in range(world)]
    dist.all_gather(emb_list, local_emb)

    # 按进程 rank 顺序拼接，得到完整 embedding
    return torch.cat(emb_list, dim=0)
