# parallel/dataloader_mp.py

import torch
from torch.utils.data import DataLoader
from core.data.datasets import InteractionDataset
from core.sampler.collate_fn import collate_subgraphs
from core.sampler.neighbor_sampler import NeighborSampler


def create_mp_dataloader(
        parquet_path,
        num_items,
        edge_list,
        num_neighbors,
        batch_size=1024,
        num_workers=4
):

    dataset = InteractionDataset(parquet_path, num_items)

    def worker_init(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        worker_info.sampler = NeighborSampler(edge_list, num_neighbors=num_neighbors)

    def _collate(batch):
        sampler = torch.utils.data.get_worker_info().sampler
        processed = []
        for (u, pos, neg) in batch:
            nodes, node2idx = sampler.sample([u, pos, neg])
            processed.append(((nodes, node2idx), (u, pos, neg)))
        return collate_subgraphs(processed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        worker_init_fn=worker_init,
        persistent_workers=True,
    )
