# core/train/eval.py

import numpy as np
import torch


def evaluate(model, all_emb, test_df, K=10, num_neg=100):
    hits = []
    ndcgs = []

    for row in test_df.itertuples():
        u = row.uid
        pos = row.iid

        # 采负例
        neg_items = np.random.randint(
            model.num_users, model.num_nodes, size=num_neg
        )
        candidates = np.concatenate([[pos], neg_items])

        u_emb = all_emb[u]
        cand_emb = all_emb[candidates]
        scores = (u_emb * cand_emb).sum(-1)

        topk_idx = torch.topk(scores, K).indices.cpu().numpy()

        # HR@K
        hit = int(0 in topk_idx)
        hits.append(hit)

        # NDCG
        if hit:
            rank = np.where(topk_idx == 0)[0][0]
            ndcg = 1.0 / np.log2(rank + 2)
        else:
            ndcg = 0.0
        ndcgs.append(ndcg)

    return float(np.mean(hits)), float(np.mean(ndcgs))
