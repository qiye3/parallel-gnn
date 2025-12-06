# core/train/train_baseline.py
# --------------------------------
# Baseline / Full-graph Forward + Batch-wise Training
# 支持反向传播与参数更新版本
# 支持可选的详细性能指标记录

import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Optional


def train_baseline(model, dataloader, edge_index,
                   optimizer, device="cpu", epochs=1, 
                   record_metrics=False) -> Optional[Dict]:
    """
    Baseline 训练函数：全图前向 + 批次训练
    
    参数:
        model: GraphSAGERecommender 模型
        dataloader: 数据加载器
        edge_index: 边索引 [2, E]
        optimizer: 优化器
        device: 设备 ("cpu" 或 "cuda")
        epochs: 训练轮数
        record_metrics: 是否记录详细性能指标
    
    返回:
        如果 record_metrics=True，返回包含指标的字典：
            - epoch_times: List[float] - 每个epoch的总时间
            - forward_times: List[float] - 每个epoch的前向传播时间
            - sampling_times: List[float] - 每个epoch的采样时间（估算）
            - losses: List[List[float]] - 每个epoch的batch损失列表
            - avg_losses: List[float] - 每个epoch的平均损失
        否则返回 None
    """
    loss_fn = nn.BCEWithLogitsLoss()
    model.to(device)
    edge_index = edge_index.to(device)

    # 初始化指标记录（如果需要）
    metrics = None
    if record_metrics:
        metrics = {
            "epoch_times": [],
            "forward_times": [],
            "sampling_times": [],
            "losses": [],
            "avg_losses": [],
        }

    for epoch in range(epochs):
        epoch_start = time.time() if record_metrics else None
        print(f"[train] ===== Epoch {epoch}/{epochs - 1} =====")

        # --------------------------------------
        # 1) 全图前向（依然关闭梯度）
        #    baseline 模式下，我们不存图，不让它反向
        # --------------------------------------
        forward_start = time.time()
        model.eval()
        with torch.no_grad():
            all_emb = model.forward_full(edge_index)   # [N, d]
        forward_time = time.time() - forward_start
        print(f"[train] 全图前向用时 {forward_time:.2f}s")
        
        if record_metrics:
            metrics["forward_times"].append(forward_time)

        # --------------------------------------
        # 2) 进入训练阶段
        # --------------------------------------
        model.train()

        total_loss = 0.0
        total_steps = 0
        epoch_losses = [] if record_metrics else None
        
        # 采样时间估算（仅当需要记录指标时）
        sampling_time_estimate = 0.0
        first_batch = record_metrics

        for step, (nodes_list, maps_list, triples) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch}")
        ):
            # 估算采样时间（第一个batch，仅当需要记录指标时）
            if first_batch:
                sampling_start = time.time()
            
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

            batch_loss_val = batch_loss.item()
            total_loss += batch_loss_val
            total_steps += 1
            
            if record_metrics:
                epoch_losses.append(batch_loss_val)
                # 估算采样时间
                if first_batch:
                    sampling_time_estimate = time.time() - sampling_start
                    first_batch = False

        avg_loss = total_loss / total_steps
        print(f"[train] Epoch {epoch} 训练平均损失 = {avg_loss:.4f}")
        
        if record_metrics:
            # 估算总采样时间（基于第一个batch的采样时间）
            total_sampling_time = sampling_time_estimate * total_steps
            metrics["sampling_times"].append(total_sampling_time)
            metrics["losses"].append(epoch_losses)
            metrics["avg_losses"].append(avg_loss)
            
            epoch_time = time.time() - epoch_start
            metrics["epoch_times"].append(epoch_time)
            print(f"[train] Epoch {epoch} 总用时 = {epoch_time:.2f}s")

    return metrics if record_metrics else None


# 为了向后兼容，保留 train_with_metrics 作为别名
def train_with_metrics(model, dataloader, edge_index,
                      optimizer, device="cpu", epochs=1) -> Dict:
    """
    带详细性能指标记录的训练函数（向后兼容别名）。
    
    此函数是 train_baseline(..., record_metrics=True) 的别名。
    """
    return train_baseline(model, dataloader, edge_index, optimizer, 
                         device=device, epochs=epochs, record_metrics=True)
