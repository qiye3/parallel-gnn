# core/train/setup.py
# --------------------
# 训练设置的公共代码，用于减少主训练脚本的重复

import torch
from core.config import Config
from core.loader.utils_io import load_graph
from core.loader.datasets import InteractionDataset
from core.sampler.neighbor_sampler import NeighborSampler
from core.sampler.collate_fn import collate_subgraphs
from parallel.dataloader_mp import create_mp_dataloader
from core.models.graphsage import GraphSAGERecommender


def create_dataloader(parquet_path, num_items, edge_list, cfg, 
                     use_mp=False, num_workers=4):
    """
    统一的 DataLoader 创建函数。
    
    参数:
        parquet_path: 训练数据路径
        num_items: 物品总数
        edge_list: 边列表
        cfg: 配置对象
        use_mp: 是否使用多进程 DataLoader
        num_workers: worker 数量（仅当 use_mp=True 时有效）
    
    返回:
        DataLoader 对象
    """
    if use_mp:
        return create_mp_dataloader(
            parquet_path=parquet_path,
            num_items=num_items,
            edge_list=edge_list,
            num_neighbors=cfg.num_neighbors,
            batch_size=cfg.batch_size,
            num_workers=num_workers,
        )
    else:
        # 单进程版本
        dataset = InteractionDataset(parquet_path, num_items)
        sampler = NeighborSampler(edge_list, cfg.num_neighbors)
        
        def _collate(batch):
            processed = []
            for u, pos, neg in batch:
                nodes, node2idx = sampler.sample([u, pos, neg])
                processed.append(((nodes, node2idx), (u, pos, neg)))
            return collate_subgraphs(processed)
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=_collate,
        )


def setup_training(cfg, use_mp=False, num_workers=4, 
                  parquet_path="data/processed/train.parquet"):
    """
    统一的训练设置函数，提取公共的初始化逻辑。
    
    参数:
        cfg: 配置对象
        use_mp: 是否使用多进程 DataLoader
        num_workers: worker 数量（仅当 use_mp=True 时有效）
        parquet_path: 训练数据路径
    
    返回:
        (model, loader, edge_index, optimizer, device) 元组
    """
    # 加载图结构
    edge_list, num_users, num_items = load_graph()
    
    # 创建 DataLoader
    loader = create_dataloader(
        parquet_path=parquet_path,
        num_items=num_items,
        edge_list=edge_list,
        cfg=cfg,
        use_mp=use_mp,
        num_workers=num_workers,
    )
    
    # 选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建模型
    model = GraphSAGERecommender(
        num_users,
        num_items,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    ).to(device)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # 转换边索引格式
    edge_index = torch.tensor(edge_list.T, dtype=torch.long).to(device)
    
    return model, loader, edge_index, optimizer, device, num_users, num_items, edge_list

