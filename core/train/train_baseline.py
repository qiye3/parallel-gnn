# core/train/train_baseline.py
# --------------------------------
# Baseline / Full-graph Forward + Batch-wise Training
# 支持反向传播与参数更新版本

import time
import torch
import torch.nn as nn
from tqdm import tqdm


def train_baseline(model, dataloader, edge_index,
                   optimizer, device="cpu", epochs=1):

    loss_fn = nn.BCEWithLogitsLoss()
    model.to(device)
    edge_index = edge_index.to(device)

    for epoch in range(epochs):
        print(f"[train] ===== Epoch {epoch}/{epochs - 1} =====")

        # --------------------------------------
        # 1) 全图前向（依然关闭梯度）
        #    baseline 模式下，我们不存图，不让它反向
        # --------------------------------------
        t0 = time.time()
        model.eval()
        with torch.no_grad():
            all_emb = model.forward_full(edge_index)   # [N, d]
        print(f"[train] 全图前向用时 {time.time() - t0:.2f}s")

        # --------------------------------------
        # 2) 进入训练阶段
        # --------------------------------------
        model.train()

        total_loss = 0.0
        total_steps = 0

        for step, (nodes_list, maps_list, triples) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch}")
        ):
            optimizer.zero_grad()

            # 因为 all_emb 是 no_grad 的，我们需要给它加一层使它能反向
            # 否则 embedding 不会更新
            all_emb_detached = all_emb.detach()
            all_emb_detached.requires_grad_(True)

            batch_loss = 0.0

            for (u, pos, neg) in triples:
                u_emb = all_emb_detached[u]
                pos_emb = all_emb_detached[pos]
                neg_emb = all_emb_detached[neg]

                pos_score = model.score(u_emb, pos_emb)
                neg_score = model.score(u_emb, neg_emb)

                scores = torch.stack([pos_score, neg_score])
                labels = torch.tensor([1.0, 0.0], device=device)

                batch_loss += loss_fn(scores, labels)

            batch_loss.backward()   # 🔥 反向传播
            optimizer.step()        # 🔥 更新 embedding + GraphSAGE 参数

            total_loss += batch_loss.item()
            total_steps += 1

        print(f"[train] Epoch {epoch} 训练平均损失 = {total_loss / total_steps:.4f}")
