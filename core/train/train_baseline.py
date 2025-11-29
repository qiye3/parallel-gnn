# core/train/train_baseline.py

import torch
import torch.nn as nn
from tqdm import tqdm


def train_baseline(model, dataloader, edge_index,
                   optimizer, device="cpu", epochs=3):

    loss_fn = nn.BCEWithLogitsLoss()
    model.to(device)
    edge_index = edge_index.to(device)

    for epoch in range(epochs):

        # 1. 先做一次全图 GNN → all_emb
        model.eval()
        with torch.no_grad():
            all_emb = model.forward_full(edge_index)

        total_loss = 0.0
        model.train()

        # 2. 遍历训练 batch
        for nodes_list, maps_list, triples in tqdm(dataloader,
                                                   desc=f"Epoch {epoch}"):

            loss = 0.0
            for (u, pos, neg) in triples:
                u_emb = all_emb[u]
                pos_emb = all_emb[pos]
                neg_emb = all_emb[neg]

                pos_score = model.score(u_emb, pos_emb)
                neg_score = model.score(u_emb, neg_emb)

                scores = torch.stack([pos_score, neg_score])
                labels = torch.tensor([1., 0.], device=device)

                loss += loss_fn(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch}] Loss = {total_loss:.4f}")
