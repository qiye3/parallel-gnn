# core/config.py
# --------------
# 统一管理训练 / 模型 / 评估等超参数的简单配置类。
# 注意：这里使用的是「类属性」，即 Config().hidden_dim 与 Config.hidden_dim 等价。


class Config:
    """
    全局配置类。

    使用方式示例：
        cfg = Config()
        print(cfg.hidden_dim, cfg.batch_size)
    """

    # ===== 模型相关配置 =====
    # 每个节点的嵌入维度，即 GraphSAGE / GAT 中的隐藏向量大小
    hidden_dim = 64
    # GNN 叠加的层数
    num_layers = 2
    # GraphSAGE 采样的邻居数配置，列表长度 = 层数
    # 如 [15, 10] 表示：第 1 层采 15 个邻居，第 2 层采 10 个邻居
    num_neighbors = [10, 5]

    # ===== 训练相关配置 =====
    # 学习率
    lr = 1e-3
    # 每个 batch 中的交互样本数
    # batch_size = 1024
    batch_size = 512
    # 训练的总 epoch 数
    epochs = 5

    # ===== 评估相关配置 =====
    # 评估时的 Top-K
    K = 10
    # 每个正样本对应采样的负样本数量
    num_neg = 50
