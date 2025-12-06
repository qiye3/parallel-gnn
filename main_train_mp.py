# main_train_mp.py
# -----------------
# 使用多进程数据加载的完整训练脚本。
# 与 main_baseline.py 的区别：
#   - 使用多进程 DataLoader（通过 setup_training 的 use_mp=True 参数）
#   - 多个 worker 进程并行进行邻居采样，加速数据准备阶段
#   - 适合 CPU 采样成为瓶颈的场景

import torch
from core.config import Config
from core.train.setup import setup_training
from core.train.train_baseline import train_baseline


def main():
    """
    使用多进程数据加载的完整训练主函数。

    主要职责：
        - 加载图结构与训练数据
        - 构建多进程数据加载器（并行采样）
        - 初始化 GraphSAGE 模型与优化器
        - 调用 train_baseline 完成训练循环
    """
    cfg = Config()
    
    # num_workers 建议设置为 CPU 核心数的一半到全部
    num_workers = 4  # 可以根据机器性能调整
    
    print(f"[main_train_mp] 初始化训练环境（多进程 DataLoader, num_workers={num_workers}）...")
    model, loader, edge_index, optimizer, device, num_users, num_items, edge_list = setup_training(
        cfg, use_mp=True, num_workers=num_workers
    )
    
    print(f"[main_train_mp] 图信息: 边数 = {edge_list.shape[0]}, 节点数 = {num_users + num_items} "
          f"(用户数 = {num_users}, 物品数 = {num_items})")
    print(f"[main_train_mp] 训练样本数 = {len(loader.dataset)}")
    print(f"[main_train_mp] 使用设备: {device}")
    print(f"[main_train_mp] 开始训练: epochs = {cfg.epochs}")
    
    train_baseline(
        model,
        loader,
        edge_index,
        optimizer,
        device=device,
        epochs=cfg.epochs,
    )
    print("[main_train_mp] 训练结束。")


if __name__ == "__main__":
    # 作为脚本直接运行时，启动多进程数据加载训练
    main()

