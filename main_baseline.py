# main_baseline.py
# -----------------
# 本文件是「基线单机训练」的入口脚本。
# 使用 core/train/setup.py 中的公共函数减少代码重复。

import torch
from core.config import Config
from core.train.setup import setup_training
from core.train.train_baseline import train_baseline


def main():
    """
    单机基线训练主函数。

    主要职责：
        - 加载图结构与训练数据
        - 构建邻居采样器与 DataLoader
        - 初始化 GraphSAGE 模型与优化器
        - 调用 train_baseline 完成训练循环
    """
    cfg = Config()
    
    print("[main] 初始化训练环境（单进程 DataLoader）...")
    model, loader, edge_index, optimizer, device, num_users, num_items, edge_list = setup_training(
        cfg, use_mp=False
    )
    
    print(f"[main] 图信息: 边数 = {edge_list.shape[0]}, 节点数 = {num_users + num_items} "
          f"(用户数 = {num_users}, 物品数 = {num_items})")
    print(f"[main] 训练样本数 = {len(loader.dataset)}")
    print(f"[main] 使用设备: {device}")
    print(f"[main] 开始训练: epochs = {cfg.epochs}")
    
    train_baseline(
        model,
        loader,
        edge_index,
        optimizer,
        device=device,
        epochs=cfg.epochs,
    )
    print("[main] 训练结束。")


if __name__ == "__main__":
    # 作为脚本直接运行时，启动基线训练
    main()
