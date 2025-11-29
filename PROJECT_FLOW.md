# 并行图神经网络推荐系统 - 完整流程文档

## 项目概述

本项目实现了一个基于 GraphSAGE 的推荐系统，支持多种并行化策略，包括：
- 多进程数据加载（Multi-Process DataLoader）
- 分布式数据并行训练（DDP - Distributed Data Parallel）
- 分布式推理（Distributed Inference）

## 项目结构

```
parallel-gnn/
├── core/                    # 核心模块
│   ├── config.py           # 配置类
│   ├── loader/              # 数据加载
│   │   ├── build_graph.py  # 图构建
│   │   ├── datasets.py     # 数据集类
│   │   └── utils_io.py     # IO工具
│   ├── models/             # 模型定义
│   │   ├── graphsage.py    # GraphSAGE模型
│   │   ├── gat.py          # GAT模型（可选）
│   │   └── layers.py       # 图神经网络层
│   ├── sampler/            # 采样器
│   │   ├── neighbor_sampler.py  # 邻居采样器
│   │   └── collate_fn.py   # 批处理函数
│   └── train/              # 训练相关
│       ├── train_baseline.py  # 基线训练
│       └── eval.py         # 评估函数
├── parallel/               # 并行化模块
│   ├── dataloader_mp.py    # 多进程数据加载器
│   ├── train_ddp.py        # DDP训练
│   ├── dist_inference.py   # 分布式推理
│   ├── graph_partition.py  # 图分区
│   └── profiler.py         # 性能分析
├── data/                   # 数据目录
│   ├── raw/                # 原始数据
│   ├── processed/          # 处理后数据
│   └── graph/              # 图数据
├── scripts/                # 运行脚本
│   ├── run_build_graph.py  # 构建图脚本
│   ├── run_baseline.sh     # 基线训练脚本
│   ├── run_ddp_2gpu.sh    # 2 GPU DDP训练
│   └── run_ddp_4gpu.sh    # 4 GPU DDP训练
├── experiments/            # 实验记录
└── main_baseline.py        # 基线训练入口

```

## 完整流程

### 阶段1: 数据准备与图构建

#### 1.1 数据预处理
**目标**: 将原始 MovieLens-32M 数据转换为图结构

**执行步骤**:
1. 读取原始评分数据 (`data/raw/ratings.csv`)
2. 扫描所有用户ID和物品ID，建立映射关系
3. 重新映射ID（用户: 0 ~ num_users-1, 物品: num_users ~ num_users+num_items-1）
4. 按时间戳进行时序划分：训练集/验证集/测试集
5. 保存处理后的数据为 Parquet 格式

**关键文件**:
- `core/loader/build_graph.py`: `build_user_item_graph()`
- `scripts/run_build_graph.py`: 执行脚本

**输出文件**:
- `data/processed/train.parquet`: 训练集交互数据
- `data/processed/val.parquet`: 验证集交互数据
- `data/processed/test.parquet`: 测试集交互数据
- `data/graph/edge_list.npy`: 边列表（用于构建图）
- `data/graph/num_info.npy`: 用户数和物品数信息

**执行命令**:
```bash
python scripts/run_build_graph.py
```

---

### 阶段2: 模型初始化

#### 2.1 配置加载
**目标**: 加载训练配置参数

**配置参数** (`core/config.py`):
- `hidden_dim = 64`: 隐藏层维度
- `num_layers = 2`: GNN层数
- `num_neighbors = [15, 10]`: 每层采样邻居数
- `lr = 1e-3`: 学习率
- `batch_size = 1024`: 批次大小
- `epochs = 4`: 训练轮数
- `K = 10`: Top-K评估
- `num_neg = 100`: 负采样数量

#### 2.2 模型构建
**目标**: 初始化 GraphSAGE 推荐模型

**模型结构** (`core/models/graphsage.py`):
- 节点嵌入层: `nn.Embedding(num_nodes, hidden_dim)`
- GraphSAGE 层: 多层图卷积网络
- 评分函数: 内积计算用户-物品相似度

**关键组件**:
- `GraphSAGERecommender`: 主模型类
- `forward_full()`: 全图前向传播，生成所有节点嵌入
- `score()`: 计算用户-物品评分

---

### 阶段3: 数据加载与采样

#### 3.1 数据集创建
**目标**: 创建交互数据集

**数据集类** (`core/loader/datasets.py`):
- `InteractionDataset`: 从 Parquet 文件加载用户-物品交互
- 每个样本返回: `(用户ID, 正样本物品ID, 负样本物品ID)`

#### 3.2 邻居采样
**目标**: 为每个训练样本采样子图

**采样器** (`core/sampler/neighbor_sampler.py`):
- `NeighborSampler`: 实现 GraphSAGE 风格的邻居采样
- 多层采样: 从种子节点开始，逐层采样邻居
- 返回: 子图节点列表和节点ID到索引的映射

**采样流程**:
1. 输入种子节点: `[用户ID, 正样本ID, 负样本ID]`
2. 第1层: 从种子节点采样15个邻居
3. 第2层: 从第1层节点采样10个邻居
4. 合并所有层节点，构建子图

#### 3.3 批处理
**目标**: 将多个样本组织成批次

**批处理函数** (`core/sampler/collate_fn.py`):
- `collate_subgraphs()`: 合并多个子图为一个批次
- 处理节点映射和边索引

---

### 阶段4: 训练流程

#### 4.1 基线训练 (Baseline Training)

**训练入口**: `main_baseline.py`

**训练流程** (`core/train/train_baseline.py`):

```
For each epoch:
  1. 模型评估模式
     - 全图前向传播: forward_full(edge_index)
     - 生成所有节点嵌入: all_emb [num_nodes, hidden_dim]
  
  2. 模型训练模式
    For each batch:
      a. 从批次中提取 (用户, 正样本, 负样本) 三元组
      b. 从 all_emb 中索引对应的嵌入向量
      c. 计算正样本和负样本的评分
      d. 使用 BCEWithLogitsLoss 计算损失
      e. 反向传播和参数更新
```

**关键特点**:
- 每个epoch开始时进行一次全图嵌入计算
- 训练时直接使用预计算的嵌入，避免重复计算
- 使用二元交叉熵损失进行优化

**执行命令**:
```bash
bash scripts/run_baseline.sh
# 或
python main_baseline.py
```

#### 4.2 多进程数据加载训练

**数据加载器** (`parallel/dataloader_mp.py`):
- `create_mp_dataloader()`: 创建多进程数据加载器
- 每个worker进程拥有独立的采样器实例
- 使用 `persistent_workers=True` 保持worker进程存活

**优势**:
- 并行采样，加速数据准备
- 减少主进程负担

#### 4.3 分布式数据并行训练 (DDP)

**训练入口**: `parallel/train_ddp.py`

**DDP训练流程**:

```
1. 初始化进程组
   - 使用 NCCL (GPU) 或 Gloo (CPU) 后端
   - 设置 rank 和 world_size

2. 数据并行化
   - 使用 DistributedSampler 划分数据集
   - 每个进程处理不同的数据子集

3. 模型并行化
   - 使用 DistributedDataParallel (DDP) 包装模型
   - 自动同步梯度

4. 训练循环
   - 每个进程独立前向传播
   - DDP自动聚合梯度
   - 同步更新参数
```

**执行命令**:
```bash
# 2 GPU训练
bash scripts/run_ddp_2gpu.sh
# 或
torchrun --nproc_per_node=2 -m parallel.train_ddp

# 4 GPU训练
bash scripts/run_ddp_4gpu.sh
# 或
torchrun --nproc_per_node=4 -m parallel.train_ddp
```

---

### 阶段5: 评估流程

#### 5.1 评估指标
**评估函数** (`core/train/eval.py`):
- `evaluate()`: 计算推荐系统评估指标

**评估指标**:
1. **HR@K (Hit Rate)**: Top-K命中率
   - 正样本是否出现在Top-K推荐列表中
   
2. **NDCG@K (Normalized Discounted Cumulative Gain)**: 归一化折损累积增益
   - 考虑排序位置的评估指标
   - 公式: `NDCG = 1 / log2(rank + 2)` (如果命中)

**评估流程**:
```
For each test sample (用户, 正样本):
  1. 采样 num_neg 个负样本
  2. 构建候选列表: [正样本, 负样本1, ..., 负样本N]
  3. 计算用户嵌入与所有候选物品嵌入的相似度
  4. 选择Top-K物品
  5. 检查正样本是否在Top-K中
  6. 计算HR和NDCG
```

#### 5.2 分布式推理
**推理模块** (`parallel/dist_inference.py`):
- `distributed_inference()`: 分布式计算节点嵌入
- 每个进程负责计算部分节点的嵌入
- 使用 `all_gather` 收集所有嵌入

---

### 阶段6: 并行化策略

#### 6.1 图分区
**分区模块** (`parallel/graph_partition.py`):
- `partition_nodes()`: 将节点划分为多个分区
- `build_subgraph()`: 为每个分区构建子图

#### 6.2 性能分析
**分析工具** (`parallel/profiler.py`):
- 性能分析和瓶颈识别

---

## 数据流

### 训练数据流

```
原始数据 (ratings.csv)
    ↓
[数据预处理]
    ↓
训练集 (train.parquet) + 图结构 (edge_list.npy)
    ↓
[数据加载]
    ↓
InteractionDataset → (用户, 正样本, 负样本)
    ↓
[邻居采样]
    ↓
NeighborSampler → 子图 (节点列表, 节点映射)
    ↓
[批处理]
    ↓
DataLoader → Batch (多个子图)
    ↓
[模型训练]
    ↓
GraphSAGE → 节点嵌入 → 评分 → 损失 → 反向传播
```

### 推理数据流

```
测试集 (test.parquet)
    ↓
[加载测试样本]
    ↓
(用户, 正样本)
    ↓
[全图嵌入计算]
    ↓
forward_full() → all_emb [num_nodes, hidden_dim]
    ↓
[评分计算]
    ↓
用户嵌入 × 候选物品嵌入 → 相似度分数
    ↓
[Top-K选择]
    ↓
评估指标 (HR@K, NDCG@K)
```

---

## 关键技术点

### 1. 图采样策略
- **GraphSAGE采样**: 逐层采样固定数量的邻居
- **子图构建**: 为每个训练样本构建独立的子图
- **内存优化**: 只处理必要的子图，而非全图

### 2. 训练策略
- **两阶段训练**: 
  - 阶段1: 全图嵌入计算（评估模式）
  - 阶段2: 基于嵌入的损失计算和优化（训练模式）
- **负采样**: 每个正样本对应一个随机负样本

### 3. 并行化策略
- **数据并行**: DDP实现模型参数同步
- **数据加载并行**: 多进程采样和批处理
- **推理并行**: 分布式计算节点嵌入

### 4. 内存管理
- **分块读取**: 大数据集分块处理
- **Parquet格式**: 高效的数据存储格式
- **子图采样**: 避免全图加载到内存

---

## 实验与评估

### 实验记录
- `experiments/exp_baseline.md`: 基线实验结果
- `experiments/exp_ddp_training.md`: DDP训练实验结果
- `experiments/exp_mp_sampler.py`: 多进程采样实验

### 性能指标
- 训练速度: 每秒处理的样本数
- 内存使用: 峰值内存占用
- 评估指标: HR@10, NDCG@10

---

## 使用指南

### 快速开始

1. **准备数据**
```bash
python scripts/run_build_graph.py
```

2. **基线训练**
```bash
python main_baseline.py
```

3. **DDP训练 (2 GPU)**
```bash
torchrun --nproc_per_node=2 -m parallel.train_ddp
```

### 自定义配置

修改 `core/config.py` 中的参数:
- 调整模型维度: `hidden_dim`
- 调整采样邻居数: `num_neighbors`
- 调整批次大小: `batch_size`
- 调整学习率: `lr`

---

## 总结

本项目实现了一个完整的并行图神经网络推荐系统，支持：
- ✅ 大规模数据处理（MovieLens-32M）
- ✅ GraphSAGE模型训练
- ✅ 多种并行化策略
- ✅ 完整的评估流程
- ✅ 可扩展的代码架构

通过合理的并行化设计，可以在多GPU环境下高效训练大规模推荐模型。

