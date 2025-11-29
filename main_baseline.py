# main_baseline.py

import numpy as np
import torch

from core.config import Config
from core.data.datasets import InteractionDataset
from core.data.utils_io import load_graph
from core.sampler.neighbor_sampler import NeighborSampler
from core.sampler.collate_fn import collate_subgraphs
from core.models.graphsage import GraphSAGERecommender
from core.train.train_baseline import train_baseline


def main():
    cfg = Config()
    edge_list, num_users, num_items = load_graph()
    sampler = NeighborSampler(edge_list, cfg.num_neighbors)

    dataset = InteractionDataset("data/processed/train.parquet", num_items)

    def _collate(batch):
        processed = []
        for u, pos, neg in batch:
            nodes, node2idx = sampler.sample([u, pos, neg])
            processed.append(((nodes, node2idx), (u, pos, neg)))
        return collate_subgraphs(processed)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=0,
        collate_fn=_collate
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GraphSAGERecommender(
        num_users, num_items,
        hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    edge_index = torch.tensor(edge_list.T, dtype=torch.long).to(device)

    train_baseline(model, loader, edge_index, optimizer,
                   device=device, epochs=cfg.epochs)


if __name__ == "__main__":
    main()
