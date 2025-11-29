# core/train/train_baseline.py
import time
import torch
import torch.nn as nn
from tqdm import tqdm

def train_baseline(model, dataloader, edge_index, optimizer,
                   device="cpu", epochs=3):

    loss_fn = nn.BCEWithLogitsLoss()
    model.to(device)
    edge_index = edge_index.to(device)

    for epoch in range(epochs):
        print(f"[train] ===== Epoch {epoch}/{epochs - 1} =====")

        # ----------- 全图前向（不建立计算图！）-----------
        t0 = time.time()
        model.eval()
        with torch.no_grad():                     # ⭐ 关键1：不建立图
            all_emb = model.forward_full(edge_index)
        t1 = time.time()
        print(f"[train] 全图前向用时 {t1 - t0:.2f}s")

        model.train()
        total_loss = 0.0

        # ----------- 小批量训练（只优化 embedding + MLP）-----------
        for step, (nodes_list, maps_list, triples) in enumerate(
                tqdm(dataloader, desc=f"Epoch {epoch}")):

            loss = 0.0

            for (u, pos, neg) in triples:
                u_emb = all_emb[u]
                pos_emb = all_emb[pos]
                neg_emb = all_emb[neg]

                pos_score = model.score(u_emb, pos_emb)
                neg_score = model.score(u_emb, neg_emb)

                scores = torch.stack([pos_score, neg_score])
                labels = torch.tensor([1., 0.], device=device)

                loss = loss + loss_fn(scores, labels)

            optimizer.zero_grad()

            # ⭐ 关键2：绝对不能 retain_graph=True
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

        print(f"[train] Epoch {epoch} loss = {total_loss:.4f}")
