# parallel/dist_inference.py

import torch
import torch.distributed as dist


def distributed_inference(model, edge_index):
    rank = dist.get_rank()
    world = dist.get_world_size()

    # 每个 worker 负责一段节点
    num_nodes = model.num_nodes
    part = num_nodes // world
    start = rank * part
    end = (rank + 1) * part if rank < world - 1 else num_nodes

    with torch.no_grad():
        full_emb = model.forward_full(edge_index)
        local_emb = full_emb[start:end]

    # gather 所有 embedding
    emb_list = [torch.zeros_like(local_emb) for _ in range(world)]
    dist.all_gather(emb_list, local_emb)

    return torch.cat(emb_list, dim=0)
