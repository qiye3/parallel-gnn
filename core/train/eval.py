# core/train/eval.py
# -------------------
# 推荐系统评估函数，实现 HR@K 与 NDCG@K 指标的计算。

import numpy as np
import torch


def evaluate(model, all_emb, test_df, K: int = 10, num_neg: int = 100):
    """
    在给定测试集上评估 Top-K 推荐表现。

    参数:
        model   : 训练好的推荐模型（需包含 num_users / num_nodes 等属性）
        all_emb : [num_nodes, dim] 的节点嵌入矩阵
        test_df : 测试集 DataFrame，至少包含列 [uid, iid]
        K       : Top-K 的 K
        num_neg : 每个正样本配的负样本数量

    评估过程：
        对于测试集中每一条 (u, pos) 交互：
            1. 从所有物品中随机采样 num_neg 个负样本
            2. 构建候选集合 [pos] + neg_items
            3. 计算用户 u 与所有候选物品的相似度得分
            4. 取得分最高的前 K 个下标，判断是否命中正样本（HR）
            5. 若命中，依据排名计算 NDCG
    """
    hits = []
    ndcgs = []

    for row in test_df.itertuples():
        u = row.uid
        pos = row.iid

        # 采负例：从所有物品节点中随机采样 num_neg 个作为负样本
        neg_items = np.random.randint(model.num_users, model.num_nodes, size=num_neg)
        # 候选集的第 0 个元素始终为正样本，其余为负样本
        candidates = np.concatenate([[pos], neg_items])

        # 取出用户和候选物品的嵌入
        u_emb = all_emb[u]
        cand_emb = all_emb[candidates]
        # 采用内积作为评分函数
        scores = (u_emb * cand_emb).sum(-1)

        # 取 Top-K 得分最高的候选下标（在 candidates 中的相对位置）
        topk_idx = torch.topk(scores, K).indices.cpu().numpy()

        # ===== HR@K =====
        # 若正样本（candidates 中下标 0）出现在 Top-K 下标集合中，则 hit=1，否则为 0
        hit = int(0 in topk_idx)
        hits.append(hit)

        # ===== NDCG@K =====
        if hit:
            # 找到正样本在 Top-K 结果中的排名（0-based）
            rank = np.where(topk_idx == 0)[0][0]
            ndcg = 1.0 / np.log2(rank + 2)
        else:
            ndcg = 0.0
        ndcgs.append(ndcg)

    # 返回平均 HR@K 与 NDCG@K
    return float(np.mean(hits)), float(np.mean(ndcgs))
