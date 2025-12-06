import time
import torch
import torch.nn as nn
from tqdm import tqdm


def train_mini_batch(model, dataloader, optimizer, device="cuda", epochs=1):
    """
    真正的 GraphSAGE 训练：
        - 每个 batch 基于邻居采样构建子图
        - 在子图上前向、计算三元组损失
        - 反向传播 + 参数更新
    """

    loss_fn = nn.BCEWithLogitsLoss()
    model.to(device)

    for epoch in range(epochs):
        print(f"\n[train] ===== Epoch {epoch}/{epochs - 1} =====")
        model.train()

        total_loss = 0.0
        total_steps = 0

        for nodes_list, maps_list, triples in tqdm(dataloader, desc=f"Epoch {epoch}"):

            # nodes_list: [batch_size 个样本，每个样本子图节点列表]
            # maps_list : 每个样本的 node2idx
            # triples   : [(u, pos, neg), ...]

            # 合并 batch 的所有子图节点（已经在 collate_subgraphs 处理好）
            nodes = nodes_list.to(device)  # [subgraph_nodes]
            triples = triples.to(device)    # [batch, 3]

            # ---- 1. 前向传播 ----
            optimizer.zero_grad()
            emb = model.forward_subgraph(nodes)  # [|subgraph|, hidden_dim]

            batch_loss = 0.0
            for (u, pos, neg) in triples:
                u_idx = torch.where(nodes == u)[0][0]
                pos_idx = torch.where(nodes == pos)[0][0]
                neg_idx = torch.where(nodes == neg)[0][0]

                u_emb = emb[u_idx]
                pos_emb = emb[pos_idx]
                neg_emb = emb[neg_idx]

                pos_score = model.score(u_emb, pos_emb)
                neg_score = model.score(u_emb, neg_emb)

                scores = torch.stack([pos_score, neg_score])
                labels = torch.tensor([1.0, 0.0], device=device)

                batch_loss += loss_fn(scores, labels)

            batch_loss = batch_loss / len(triples)

            # ---- 2. 反向传播 ----
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            total_steps += 1

        avg_loss = total_loss / total_steps
        print(f"[train] Epoch {epoch} 平均训练损失 = {avg_loss:.4f}")
