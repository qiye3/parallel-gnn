# 执行流程 (EXECUTION_FLOW)

本文档详细描述项目的执行流程，包括函数调用顺序、数据流向和关键执行路径。

---

## 1. 数据准备流程

### 1.1 图构建执行流程

```
scripts/run_build_graph.py
    │
    └─> core/data/build_graph.py::build_user_item_graph()
            │
            ├─> remap_ids_full()
            │       │
            │       └─> load_movielens_32m_chunked()
            │               └─> pd.read_csv() [分块读取]
            │
            ├─> load_movielens_32m_chunked() [第二遍扫描]
            │       └─> pd.read_csv() [分块读取]
            │
            ├─> temporal_split()
            │       └─> df.sort_values("timestamp")
            │
            ├─> train.to_parquet("data/processed/train.parquet")
            ├─> val.to_parquet("data/processed/val.parquet")
            ├─> test.to_parquet("data/processed/test.parquet")
            │
            └─> np.save("data/graph/edge_list.npy", edge_list)
            └─> np.save("data/graph/num_info.npy", [num_users, num_items])
```

**执行顺序**:
1. 第一遍扫描: 收集所有用户ID和物品ID
2. 建立ID映射: `uid_map`, `iid_map`
3. 第二遍扫描: 重新映射ID并合并数据
4. 时序划分: 训练集/验证集/测试集
5. 保存数据: Parquet格式和Numpy格式

---

## 2. 基线训练执行流程

### 2.1 主程序入口

```
main_baseline.py::main()
    │
    ├─> Config() [加载配置]
    │
    ├─> load_graph()
    │       │
    │       └─> core/data/utils_io.py::load_graph()
    │               ├─> np.load("data/graph/edge_list.npy")
    │               └─> np.load("data/graph/num_info.npy")
    │
    ├─> NeighborSampler(edge_list, cfg.num_neighbors)
    │       │
    │       └─> core/sampler/neighbor_sampler.py::__init__()
    │               └─> 构建邻接表 self.adj
    │
    ├─> InteractionDataset("data/processed/train.parquet", num_items)
    │       │
    │       └─> core/data/datasets.py::__init__()
    │               └─> pd.read_parquet()
    │
    ├─> GraphSAGERecommender(num_users, num_items, ...)
    │       │
    │       └─> core/models/graphsage.py::__init__()
    │               ├─> nn.Embedding(num_nodes, hidden_dim)
    │               └─> nn.ModuleList([GraphSAGELayer(...)])
    │
    ├─> torch.optim.Adam(model.parameters(), lr=cfg.lr)
    │
    ├─> DataLoader(dataset, batch_size, collate_fn=_collate)
    │
    └─> train_baseline(model, loader, edge_index, optimizer, ...)
```

### 2.2 训练循环执行流程

```
core/train/train_baseline.py::train_baseline()
    │
    └─> For epoch in range(epochs):
            │
            ├─> [阶段1: 全图嵌入计算]
            │   │
            │   ├─> model.eval()
            │   ├─> torch.no_grad()
            │   └─> model.forward_full(edge_index)
            │           │
            │           └─> core/models/graphsage.py::forward_full()
            │                   │
            │                   ├─> self.emb(nodes)  [节点嵌入]
            │                   │
            │                   └─> For layer in self.layers:
            │                           └─> layer(x, edge_index)
            │                                   │
            │                                   └─> core/models/layers.py::GraphSAGELayer.forward()
            │                                           ├─> 消息聚合
            │                                           └─> 特征更新
            │
            ├─> [阶段2: 批次训练]
            │   │
            │   ├─> model.train()
            │   │
            │   └─> For batch in dataloader:
            │           │
            │           ├─> DataLoader.__iter__()
            │           │       │
            │           │       └─> InteractionDataset.__getitem__(idx)
            │           │               └─> 返回 (u, pos, neg)
            │           │
            │           ├─> _collate(batch)
            │           │       │
            │           │       ├─> For (u, pos, neg) in batch:
            │           │       │       └─> sampler.sample([u, pos, neg])
            │           │       │               │
            │           │       │               └─> core/sampler/neighbor_sampler.py::sample()
            │           │       │                       │
            │           │       │                       ├─> 第1层采样: 从种子节点采样15个邻居
            │           │       │                       ├─> 第2层采样: 从第1层节点采样10个邻居
            │           │       │                       └─> 返回 (all_nodes, node2idx)
            │           │       │
            │           │       └─> collate_subgraphs(processed)
            │           │               │
            │           │               └─> core/sampler/collate_fn.py::collate_subgraphs()
            │           │                       └─> 合并多个子图为一个批次
            │           │
            │           ├─> For (u, pos, neg) in triples:
            │           │       │
            │           │       ├─> u_emb = all_emb[u]
            │           │       ├─> pos_emb = all_emb[pos]
            │           │       ├─> neg_emb = all_emb[neg]
            │           │       │
            │           │       ├─> pos_score = model.score(u_emb, pos_emb)
            │           │       │       │
            │           │       │       └─> core/models/graphsage.py::score()
            │           │       │               └─> (u_emb * i_emb).sum(-1)
            │           │       │
            │           │       ├─> neg_score = model.score(u_emb, neg_emb)
            │           │       │
            │           │       └─> loss += loss_fn(scores, labels)
            │           │               │
            │           │               └─> nn.BCEWithLogitsLoss()
            │           │
            │           ├─> optimizer.zero_grad()
            │           ├─> loss.backward()
            │           └─> optimizer.step()
```

**关键执行路径**:
1. **每个Epoch开始**: 计算全图嵌入 `all_emb`
2. **每个Batch**: 
   - 从数据集获取 `(用户, 正样本, 负样本)`
   - 为每个三元组采样子图
   - 从 `all_emb` 中索引嵌入
   - 计算评分和损失
   - 反向传播更新参数

---

## 3. 多进程数据加载执行流程

### 3.1 多进程数据加载器创建

```
parallel/dataloader_mp.py::create_mp_dataloader()
    │
    ├─> InteractionDataset(parquet_path, num_items)
    │
    ├─> DataLoader(..., num_workers=4, worker_init_fn=worker_init)
    │       │
    │       └─> 创建4个worker进程
    │
    └─> worker_init(worker_id)
            │
            └─> 为每个worker创建独立的 NeighborSampler 实例
```

### 3.2 Worker进程执行流程

```
Worker Process (每个worker独立执行)
    │
    ├─> worker_init(worker_id)
    │       │
    │       └─> worker_info.sampler = NeighborSampler(...)
    │
    └─> _collate(batch)
            │
            ├─> torch.utils.data.get_worker_info().sampler
            │       └─> 获取当前worker的采样器
            │
            └─> For (u, pos, neg) in batch:
                    └─> sampler.sample([u, pos, neg])
                            └─> 使用worker独立的采样器进行采样
```

**并行化优势**:
- 多个worker同时进行邻居采样
- 减少主进程负担
- 提高数据加载速度

---

## 4. DDP分布式训练执行流程

### 4.1 DDP初始化流程

```
parallel/train_ddp.py::main_ddp()
    │
    ├─> rank = os.environ["RANK"]
    ├─> world = os.environ["WORLD_SIZE"]
    │
    ├─> setup(rank, world)
    │       │
    │       ├─> dist.init_process_group(backend="nccl", ...)
    │       │       └─> 初始化进程组通信
    │       │
    │       └─> torch.cuda.set_device(rank)
    │               └─> 设置当前进程的GPU设备
    │
    ├─> Config()
    ├─> load_graph()
    ├─> InteractionDataset(...)
    ├─> NeighborSampler(...)
    │
    ├─> DistributedSampler(dataset, num_replicas=world, rank=rank)
    │       │
    │       └─> 为每个进程分配不同的数据子集
    │
    ├─> GraphSAGERecommender(...)
    │
    ├─> DDP(model, device_ids=[rank])
    │       │
    │       └─> 包装模型，自动同步梯度
    │
    └─> train_baseline(ddp_model, loader, edge_index, optimizer, ...)
```

### 4.2 DDP训练循环执行流程

```
每个进程独立执行 (rank 0, 1, 2, ...)
    │
    └─> train_baseline(ddp_model, ...)
            │
            └─> For epoch in range(epochs):
                    │
                    ├─> [全图嵌入计算 - 每个进程独立]
                    │   │
                    │   └─> ddp_model.forward_full(edge_index)
                    │           │
                    │           └─> 每个进程计算完整全图嵌入
                    │
                    ├─> [数据采样 - 分布式采样]
                    │   │
                    │   └─> For batch in dataloader:
                    │           │
                    │           └─> DistributedSampler 确保每个进程
                    │                  处理不同的数据批次
                    │
                    ├─> [前向传播 - 每个进程独立]
                    │   │
                    │   └─> ddp_model(...)
                    │           │
                    │           └─> 每个进程独立计算前向传播
                    │
                    ├─> [反向传播 - 自动同步]
                    │   │
                    │   ├─> loss.backward()
                    │   │       │
                    │   │       └─> DDP自动聚合所有进程的梯度
                    │   │
                    │   └─> optimizer.step()
                    │           │
                    │           └─> 所有进程同步更新参数
```

**DDP关键机制**:
1. **数据并行**: `DistributedSampler` 划分数据
2. **梯度同步**: DDP自动聚合梯度
3. **参数同步**: 所有进程保持模型参数一致

---

## 5. 评估执行流程

### 5.1 评估流程

```
core/train/eval.py::evaluate(model, all_emb, test_df, K=10, num_neg=100)
    │
    └─> For row in test_df.itertuples():
            │
            ├─> u = row.uid
            ├─> pos = row.iid
            │
            ├─> neg_items = np.random.randint(...)
            │       └─> 采样 num_neg 个负样本
            │
            ├─> candidates = [pos] + neg_items
            │
            ├─> u_emb = all_emb[u]
            ├─> cand_emb = all_emb[candidates]
            │
            ├─> scores = (u_emb * cand_emb).sum(-1)
            │       └─> 计算用户与所有候选物品的相似度
            │
            ├─> topk_idx = torch.topk(scores, K).indices
            │       └─> 选择Top-K物品
            │
            ├─> hit = int(0 in topk_idx)
            │       └─> 检查正样本是否在Top-K中
            │
            ├─> hits.append(hit)
            │
            └─> ndcg = 1.0 / np.log2(rank + 2) if hit else 0.0
                    └─> 计算NDCG指标
```

---

## 6. 分布式推理执行流程

### 6.1 分布式推理流程

```
parallel/dist_inference.py::distributed_inference(model, edge_index)
    │
    ├─> rank = dist.get_rank()
    ├─> world = dist.get_world_size()
    │
    ├─> num_nodes = model.num_nodes
    ├─> part = num_nodes // world
    ├─> start = rank * part
    ├─> end = (rank + 1) * part
    │       └─> 计算当前进程负责的节点范围
    │
    ├─> model.forward_full(edge_index)
    │       └─> 每个进程计算完整全图嵌入
    │
    ├─> local_emb = full_emb[start:end]
    │       └─> 提取当前进程负责的节点嵌入
    │
    ├─> emb_list = [torch.zeros_like(local_emb) for _ in range(world)]
    │
    └─> dist.all_gather(emb_list, local_emb)
            │
            └─> 收集所有进程的嵌入
            │
            └─> return torch.cat(emb_list, dim=0)
                    └─> 拼接为完整嵌入矩阵
```

**分布式推理优势**:
- 每个进程只负责部分节点的嵌入计算
- 使用 `all_gather` 收集所有嵌入
- 减少单进程内存压力

---

## 7. 完整执行时序图

### 7.1 基线训练完整时序

```
时间轴 →
│
├─ [初始化阶段]
│   ├─> 加载配置
│   ├─> 加载图数据
│   ├─> 创建采样器
│   ├─> 创建数据集
│   ├─> 创建模型
│   └─> 创建优化器
│
├─ [训练阶段 - Epoch 0]
│   ├─> [全图嵌入计算]
│   │   └─> forward_full() → all_emb
│   │
│   └─> [批次训练]
│       ├─> Batch 0: 采样 → 计算损失 → 反向传播
│       ├─> Batch 1: 采样 → 计算损失 → 反向传播
│       └─> ...
│
├─ [训练阶段 - Epoch 1]
│   ├─> [全图嵌入计算]
│   └─> [批次训练]
│
└─ ...
```

### 7.2 DDP训练完整时序

```
时间轴 →
│
├─ [初始化阶段 - 所有进程]
│   ├─> 初始化进程组
│   ├─> 设置GPU设备
│   ├─> 加载数据
│   ├─> 创建分布式采样器
│   ├─> 创建模型
│   └─> 包装为DDP模型
│
├─ [训练阶段 - Epoch 0]
│   ├─> [全图嵌入计算 - 所有进程独立]
│   │
│   └─> [批次训练 - 并行执行]
│       ├─> Process 0: Batch 0, 2, 4, ...
│       ├─> Process 1: Batch 1, 3, 5, ...
│       └─> ...
│       │
│       └─> [梯度同步 - 自动]
│           └─> DDP聚合所有进程的梯度
│
└─ ...
```

---

## 8. 关键函数调用链

### 8.1 邻居采样调用链

```
NeighborSampler.sample([u, pos, neg])
    │
    ├─> layers = [seeds]  # 初始化种子层
    ├─> frontier = seeds
    │
    └─> For k in num_neighbors:  # [15, 10]
            │
            ├─> For nid in frontier:
            │       │
            │       ├─> neigh = self.adj[nid]  # 获取邻居
            │       │
            │       └─> sampled = np.random.choice(neigh, k)
            │               └─> 采样k个邻居
            │
            └─> frontier = 新采样的节点集合
```

### 8.2 模型前向传播调用链

```
GraphSAGERecommender.forward_full(edge_index)
    │
    ├─> nodes = torch.arange(num_nodes)
    ├─> x = self.emb(nodes)  # [num_nodes, hidden_dim]
    │
    └─> For layer in self.layers:
            │
            └─> GraphSAGELayer.forward(x, edge_index)
                    │
                    ├─> 消息聚合 (从邻居节点)
                    ├─> 特征更新
                    └─> 返回更新后的 x
```

---

## 9. 数据流向图

### 9.1 训练数据流

```
原始数据
    ↓
[build_graph.py]
    ↓
train.parquet + edge_list.npy
    ↓
[InteractionDataset]
    ↓
(u, pos, neg) 三元组
    ↓
[NeighborSampler]
    ↓
子图 (nodes, node2idx)
    ↓
[collate_subgraphs]
    ↓
批次数据
    ↓
[GraphSAGE模型]
    ↓
节点嵌入
    ↓
[评分计算]
    ↓
损失值
    ↓
[反向传播]
    ↓
参数更新
```

### 9.2 评估数据流

```
test.parquet
    ↓
[加载测试样本]
    ↓
(u, pos) 对
    ↓
[全图嵌入 all_emb]
    ↓
用户嵌入 + 候选物品嵌入
    ↓
[相似度计算]
    ↓
评分分数
    ↓
[Top-K选择]
    ↓
HR@K, NDCG@K
```

---

## 10. 执行模式对比

| 模式 | 数据加载 | 采样 | 前向传播 | 梯度同步 | 适用场景 |
|------|---------|------|---------|---------|---------|
| **Baseline** | 单进程 | 串行 | 单GPU | 无 | 小规模数据，单GPU |
| **MP DataLoader** | 多进程 | 并行采样 | 单GPU | 无 | 中等规模，单GPU，需要加速采样 |
| **DDP** | 分布式采样 | 并行采样 | 多GPU | 自动同步 | 大规模数据，多GPU |

---

## 总结

本项目的执行流程遵循以下原则：

1. **模块化设计**: 每个模块职责清晰，易于扩展
2. **并行化策略**: 支持多种并行化方式，适应不同场景
3. **内存优化**: 使用采样和分块处理，避免全图加载
4. **灵活配置**: 通过Config类统一管理参数

通过理解这些执行流程，可以更好地：
- 调试和优化代码
- 扩展新功能
- 理解性能瓶颈
- 选择合适的并行化策略

