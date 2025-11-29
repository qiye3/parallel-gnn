# parallel/train_ddp.py

import os
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from core.data.datasets import InteractionDataset
from core.sampler.collate_fn import collate_subgraphs
from core.sampler.neighbor_sampler import NeighborSampler
from core.models.graphsage import GraphSAGERecommender
from core.train.train_baseline import train_baseline
from core.config import Config


def setup(rank, world):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend,
                            init_method="env://",
                            rank=rank, world_size=world)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def main_ddp():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    setup(rank, world)

    cfg = Config()
    edge_list = np.load("data/graph/edge_list.npy")
    num_users, num_items = np.load("data/graph/num_info.npy")

    dataset = InteractionDataset("data/processed/train.parquet", num_items)
    sampler = NeighborSampler(edge_list, cfg.num_neighbors)

    dist_sampler = DistributedSampler(dataset, num_replicas=world, rank=rank)

    def _collate(batch):
        processed = []
        for u, pos, neg in batch:
            nodes, node2idx = sampler.sample([u, pos, neg])
            processed.append(((nodes, node2idx), (u, pos, neg)))
        return collate_subgraphs(processed)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=dist_sampler,
        num_workers=2,
        collate_fn=_collate
    )

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    model = GraphSAGERecommender(
        num_users, num_items, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers
    ).to(device)

    ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=cfg.lr)

    edge_index = torch.tensor(edge_list.T, dtype=torch.long).to(device)

    train_baseline(ddp_model, loader, edge_index, optimizer,
                   device=device, epochs=cfg.epochs)

    dist.destroy_process_group()
