# core/train/train_mini_batch.py
# --------------------------------
# Mini-batch GraphSAGE 训练函数：真正的子图级别训练实现。
# 与 baseline 训练（train_baseline.py）的主要区别：
#   - baseline: 每个 epoch 先做一次全图前向，然后基于预计算的 embedding 计算损失
#   - mini-batch: 每个 batch 动态构建子图，在子图上执行 GNN 前向和反向传播
# 优势：显存占用更可控，适合大规模图训练；劣势：每个 batch 都需要重新采样和构建子图

import torch
import torch.nn as nn
from tqdm import tqdm

from core.sampler.neighbor_sampler import NeighborSampler


def train_mini_batch_gnn(
    model,
    dataloader,
    sampler: NeighborSampler,
    optimizer,
    device: str = "cpu",
    epochs: int = 3,
):
    """
    Mini-batch GraphSAGE 训练主函数。

    训练流程（每个 batch）：
        1. 从 DataLoader 获取 (u_batch, pos_batch, neg_batch) 三元组
        2. 以 batch 中的所有节点（用户、正样本、负样本）作为种子节点
        3. 使用 NeighborSampler 从种子节点出发，采样多层邻居，构建子图
        4. 在子图上执行 GNN 前向传播（forward_subgraph），得到子图节点的嵌入
        5. 从子图嵌入中索引出用户、正样本、负样本的嵌入，计算对比损失
        6. 反向传播更新模型参数

    参数:
        model     : GraphSAGERecommender 模型实例
        dataloader: PyTorch DataLoader，返回 (u_batch, pos_batch, neg_batch) 三个张量
                   注意：这里使用默认的 collate_fn，而不是 baseline 中的自定义 collate_fn
        sampler   : NeighborSampler 实例，用于为每个 batch 构建子图
        optimizer : 优化器（如 Adam）
        device    : 训练设备（"cpu" 或 "cuda"）
        epochs    : 训练轮数
    """
    # 使用二元交叉熵损失（带 logits），用于正负样本对比学习
    loss_fn = nn.BCEWithLogitsLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()  # 设置为训练模式（启用 dropout 等）
        total_loss = 0.0
        total_triples = 0

        print(f"[mini-batch] ===== Epoch {epoch}/{epochs - 1} =====")

        # 遍历所有 batch
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            # DataLoader 默认会把多个 (u, pos, neg) 三元组堆叠成三个张量
            # u_batch: [batch_size]，pos_batch: [batch_size]，neg_batch: [batch_size]
            u_batch, pos_batch, neg_batch = batch
            # 转换为 Python 列表，便于后续处理
            u_batch = u_batch.tolist()
            pos_batch = pos_batch.tolist()
            neg_batch = neg_batch.tolist()

            # ===== 步骤 1: 收集种子节点 =====
            # 将当前 batch 中的所有用户、正样本、负样本节点作为种子节点
            # 使用 set 去重，避免重复采样
            seeds = list(set(u_batch + pos_batch + neg_batch))

            # ===== 步骤 2: 构建子图 =====
            # 从种子节点出发，按照 num_neighbors 配置采样多层邻居
            # 返回：
            #   - sub_nodes: 子图中所有节点的全局 ID 列表
            #   - node2idx: 全局 ID 到子图局部索引的映射字典
            #   - edge_sub: [E_sub, 2] 形状的子图边列表（使用局部索引）
            sub_nodes, node2idx, edge_sub = sampler.sample_subgraph(seeds)
            # 极端情况：如果子图没有边（所有种子节点都是孤立节点），跳过该 batch
            if edge_sub.shape[0] == 0:
                continue

            # 将子图信息转换为 PyTorch 张量并移动到指定设备
            global_ids = torch.tensor(sub_nodes, dtype=torch.long, device=device)
            # edge_sub 是 [E_sub, 2] 形状，需要转置为 [2, E_sub] 格式
            edge_index_sub = torch.tensor(edge_sub.T, dtype=torch.long, device=device)

            # ===== 步骤 3: 在子图上执行 GNN 前向传播 =====
            # forward_subgraph 会根据全局节点 ID 获取嵌入，然后在子图上执行卷积
            # 返回子图所有节点的嵌入向量
            h_sub = model.forward_subgraph(global_ids, edge_index_sub)  # [N_sub, dim]

            # ===== 步骤 4: 计算对比损失 =====
            loss = 0.0
            B = len(u_batch)  # batch 大小
            for i in range(B):
                u = u_batch[i]      # 用户节点全局 ID
                pos = pos_batch[i]  # 正样本节点全局 ID
                neg = neg_batch[i]  # 负样本节点全局 ID

                # 检查三个节点是否都在子图中（理论上应该都在，因为种子节点包含了它们）
                # 如果不在，跳过该样本（可能是采样过程中的边界情况）
                if (u not in node2idx) or (pos not in node2idx) or (neg not in node2idx):
                    continue

                # 将全局 ID 转换为子图局部索引
                u_idx = node2idx[u]
                pos_idx = node2idx[pos]
                neg_idx = node2idx[neg]

                # 从子图嵌入中索引出对应的嵌入向量
                u_emb = h_sub[u_idx]
                pos_emb = h_sub[pos_idx]
                neg_emb = h_sub[neg_idx]

                # 计算用户与正样本、负样本的相似度得分
                pos_score = model.score(u_emb, pos_emb)
                neg_score = model.score(u_emb, neg_emb)

                # 将正负样本得分堆叠，标签为 [1, 0]（希望正样本得分高，负样本得分低）
                scores = torch.stack([pos_score, neg_score])
                labels = torch.tensor([1.0, 0.0], device=device)
                # 累加该样本的损失
                loss = loss + loss_fn(scores, labels)

            # 如果该 batch 的所有样本都被跳过（loss 仍为 0），则跳过反向传播
            if loss == 0.0:
                continue

            # ===== 步骤 5: 反向传播与参数更新 =====
            optimizer.zero_grad()  # 清零梯度
            loss.backward()        # 反向传播计算梯度
            optimizer.step()       # 更新模型参数

            # 累计损失和样本数，用于计算 epoch 平均损失
            total_loss += loss.item()
            total_triples += B

        # 计算并打印该 epoch 的平均损失
        avg_loss = total_loss / max(total_triples, 1)
        print(f"[mini-batch] Epoch {epoch} 平均损失 = {avg_loss:.4f}")
