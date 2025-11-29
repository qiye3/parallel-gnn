# core/train/train_baseline.py
# --------------------------------
# Baseline / quick smoke test 版本：
#  - 每个 epoch：全图前向一次，得到 all_emb
#  - 使用 all_emb 计算 batch 上的“伪损失”，仅用于观察，不做反向传播

import time
import torch
import torch.nn as nn
from tqdm import tqdm


def train_baseline(model, dataloader, edge_index,
                   optimizer, device: str = "cpu", epochs: int = 1):
    """
    用于 baseline / quick_train 的“只前向、不反向”版本。

    作用：
      - 验证数据管道 + 采样 + 模型前向是否正常
      - 不更新模型参数（不 backward，不 step）

    参数:
        model      : GraphSAGERecommender
        dataloader : 输出 (nodes_list, maps_list, triples) 的 DataLoader
        edge_index : [2, E] 整图边
        optimizer  : 这里不会真的用到，仅为接口兼容
        device     : "cpu" or "cuda"
        epochs     : 跑几轮（一般 quick_train 设为 1 即可）
    """
    loss_fn = nn.BCEWithLogitsLoss()
    model.to(device)
    edge_index = edge_index.to(device)

    for epoch in range(epochs):
        print(f"[train] ===== Epoch {epoch}/{epochs - 1} =====")

        # -------- 全图前向（完全关闭梯度） --------
        t0 = time.time()
        model.eval()
        with torch.no_grad():
            # all_emb: [num_nodes, hidden_dim]
            all_emb = model.forward_full(edge_index)
        t1 = time.time()
        print(f"[train] 全图前向用时 {t1 - t0:.2f}s")

        total_loss = 0.0
        total_triples = 0

        # -------- 仅做 loss 统计，不做 backward --------
        for step, (nodes_list, maps_list, triples) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch}")
        ):
            batch_loss = 0.0

            for (u, pos, neg) in triples:
                u_emb = all_emb[u]
                pos_emb = all_emb[pos]
                neg_emb = all_emb[neg]

                pos_score = model.score(u_emb, pos_emb)
                neg_score = model.score(u_emb, neg_emb)

                scores = torch.stack([pos_score, neg_score])
                labels = torch.tensor([1.0, 0.0], device=device)
                batch_loss += loss_fn(scores, labels).item()

            total_loss += batch_loss
            total_triples += len(triples)

        avg_loss = total_loss / max(total_triples, 1)
        print(f"[train] Epoch {epoch} 平均“伪损失” = {avg_loss:.4f}")
