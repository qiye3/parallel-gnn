# Baseline 完整运行流程说明

本文档详细说明从读取原始数据到完成训练的完整 baseline 流程，包括每个步骤使用的代码、中间输出和文件。

---

## 阶段一：数据预处理与图构建

### 1.1 执行入口

**脚本**: `scripts/run_build_graph.py`

**执行命令**:
```bash
python scripts/run_build_graph.py
```

**代码位置**: ```1:22:scripts/run_build_graph.py```

该脚本负责调用核心的图构建函数。

---

### 1.2 读取原始数据

**代码位置**: ```12:29:core/loader/build_graph.py```

**函数**: `load_ml10m_chunked()`

**输入文件**: 
- `data/raw/ratings.dat` (MovieLens-10M 数据集)
- 数据格式: `UserID::MovieID::Rating::Timestamp` (使用 `::` 分隔符)

**处理过程**:
- 使用 pandas 按块读取（chunksize=500,000），避免内存溢出
- 指定列名: `["userId", "movieId", "rating", "timestamp"]`
- 数据类型: userId/movieId 为 int32，rating 为 float32，timestamp 为 int64

**中间输出**: 
- 返回一个 pandas DataFrame 迭代器，每次迭代返回一个 chunk

---

### 1.3 ID 重映射（Pass 1）

**代码位置**: ```32:56:core/loader/build_graph.py```

**函数**: `remap_ids_full()`

**处理过程**:
1. 扫描所有 chunks，收集所有唯一的 `userId` 和 `movieId`
2. 对用户ID和物品ID分别排序
3. 构建映射字典:
   - `uid_map`: 原始用户ID → 连续用户ID [0, num_users)
   - `iid_map`: 原始物品ID → 连续物品ID [num_users, num_users + num_items)

**输出信息**:
```
=== Pass 1: scanning all IDs (ML-10M) ===
  Users: <用户总数>
  Items: <物品总数>
```

**返回值**:
- `uid_map`: 用户ID映射字典
- `iid_map`: 物品ID映射字典
- `num_users`: 用户总数
- `num_items`: 物品总数

---

### 1.4 数据重编码与合并（Pass 2）

**代码位置**: ```92:108:core/loader/build_graph.py```

**处理过程**:
1. 重新读取所有 chunks
2. 对每个 chunk 应用 ID 映射:
   - `uid = chunk["userId"].map(uid_map)`
   - `iid = chunk["movieId"].map(iid_map)`
3. 保留 `rating` 和 `timestamp` 列
4. 将所有 chunks 合并为一个完整的 DataFrame

**输出信息**:
```
=== Pass 2: remapping IDs & concatenating ===
  Total interactions = <总交互数>
```

**中间数据**: 
- `df_all`: 包含所有交互的 DataFrame，列包括 `uid`, `iid`, `rating`, `timestamp`

---

### 1.5 数据集划分

**代码位置**: ```59:71:core/loader/build_graph.py```

**函数**: `temporal_split()`

**处理过程**:
1. 按 `timestamp` 列排序（时间顺序）
2. 按比例划分:
   - 最后 10% 作为测试集
   - 倒数 10%-20% 作为验证集
   - 前 80% 作为训练集

**输出信息**:
```
=== Splitting train/val/test ===
```

**输出文件**:
- `data/processed/train.parquet`: 训练集（包含 uid, iid, rating, timestamp）
- `data/processed/val.parquet`: 验证集
- `data/processed/test.parquet`: 测试集

---

### 1.6 构建图结构

**代码位置**: ```117:121:core/loader/build_graph.py```

**处理过程**:
1. 从训练集构建边列表:
   - 提取 `(uid, iid)` 对
   - 转换为 numpy 数组，形状为 `[E, 2]`
2. 保存图统计信息:
   - `num_users` 和 `num_items`

**输出文件**:
- `data/graph/edge_list.npy`: 边列表，形状 `[E, 2]`，数据类型 int32
- `data/graph/num_info.npy`: 数组 `[num_users, num_items]`，数据类型 int64

**输出信息**:
```
=== Building edge_list from TRAIN only ===
=== Done ===
 Users = <用户数>
 Items = <物品数>
 Train edges = <训练边数>
```

---

## 阶段二：模型训练

### 2.1 训练入口

**脚本**: `main_baseline.py`

**执行命令**:
```bash
python main_baseline.py
```

**代码位置**: ```24:111:main_baseline.py```

---

### 2.2 加载配置

**代码位置**: ```34:35:main_baseline.py```

**代码**: ```7:38:core/config.py```

**配置参数**:
- `hidden_dim = 64`: 节点嵌入维度
- `num_layers = 2`: GNN 层数
- `num_neighbors = [15, 10]`: 每层采样邻居数
- `lr = 1e-3`: 学习率
- `batch_size = 16`: 批次大小
- `epochs = 4`: 训练轮数

---

### 2.3 加载图结构

**代码位置**: ```37:41:main_baseline.py```

**函数**: ```8:25:core/loader/utils_io.py```

**处理过程**:
1. 读取 `data/graph/edge_list.npy` → `edge_list` (形状 `[E, 2]`)
2. 读取 `data/graph/num_info.npy` → `[num_users, num_items]`

**输出信息**:
```
[main] 加载图结构 ...
[main] 图加载完成: 边数 = <E>, 节点数 = <num_users + num_items> (用户数 = <num_users>, 物品数 = <num_items>)
```

**返回值**:
- `edge_list`: numpy 数组，形状 `[E, 2]`
- `num_users`: 用户总数
- `num_items`: 物品总数

---

### 2.4 构建邻居采样器

**代码位置**: ```43:45:main_baseline.py```

**代码**: ```14:27:core/sampler/neighbor_sampler.py```

**处理过程**:
1. 初始化 `NeighborSampler`，传入 `edge_list` 和 `num_neighbors`
2. 构建无向邻接表 `adj`:
   - 为每个节点维护其邻居列表
   - 边 `(u, v)` 会在 `adj[u]` 和 `adj[v]` 中都添加

**输出信息**:
```
[main] 构建 NeighborSampler, num_neighbors = [15, 10]
```

**内部数据结构**:
- `self.adj`: 邻接表，`adj[i]` 是节点 `i` 的所有邻居列表
- `self.num_nodes`: 总节点数

---

### 2.5 构建交互数据集

**代码位置**: ```47:50:main_baseline.py```

**代码**: ```11:53:core/loader/datasets.py```

**处理过程**:
1. 读取 `data/processed/train.parquet`
2. 提取 `uid` 和 `iid` 列，转换为 numpy 数组
3. 保存 `num_items` 用于负采样

**输出信息**:
```
[main] 加载训练数据集 data/processed/train.parquet ...
[main] 训练样本数 = <训练样本数>
```

**数据集行为**:
- `__getitem__(idx)`: 返回 `(u, pos, neg)` 三元组
  - `u`: 用户ID（从训练集中获取）
  - `pos`: 正样本物品ID（用户真实交互的物品）
  - `neg`: 负样本物品ID（从 `[0, num_items)` 随机采样）

---

### 2.6 构建 DataLoader

**代码位置**: ```52:75:main_baseline.py```

**处理过程**:

#### 2.6.1 自定义 Collate 函数

**代码位置**: ```53:65:main_baseline.py```

**处理流程**:
1. 对 batch 中的每个样本 `(u, pos, neg)`:
   - 调用 `sampler.sample([u, pos, neg])` 进行多层邻居采样
   - 返回 `(nodes, node2idx)`，其中:
     - `nodes`: 子图中所有节点的全局ID列表
     - `node2idx`: 全局ID → 子图局部索引的映射字典
2. 调用 `collate_subgraphs()` 合并 batch

**采样过程** (```30:50:core/sampler/neighbor_sampler.py```):
- 第1层: 从种子节点 `[u, pos, neg]` 采样 15 个邻居
- 第2层: 从第1层节点采样 10 个邻居
- 合并所有层节点，去重并排序
- 构建 `node2idx` 映射

#### 2.6.2 DataLoader 初始化

**参数**:
- `batch_size = 16`: 每个 batch 包含 16 个样本
- `shuffle = True`: 随机打乱
- `num_workers = 0`: 单进程加载
- `collate_fn = _collate`: 使用自定义 collate 函数

**输出信息**:
```
[main] 构建 DataLoader: batch_size = 16, num_workers = 0
```

**DataLoader 输出格式**:
- `nodes_list`: `List[List[int]]`，每个元素是一个样本的子图节点列表
- `maps_list`: `List[Dict[int, int]]`，每个元素是 node2idx 映射
- `triples`: `List[Tuple[int, int, int]]`，每个元素是 `(u, pos, neg)` 三元组

---

### 2.7 初始化模型

**代码位置**: ```77:88:main_baseline.py```

**代码**: ```11:51:core/models/graphsage.py```

**处理过程**:
1. 选择设备（优先 GPU）
2. 初始化 `GraphSAGERecommender`:
   - 创建嵌入层: `nn.Embedding(num_nodes, hidden_dim)`
   - 创建 `num_layers` 个 `GraphSAGELayer`
3. 将模型移动到设备

**输出信息**:
```
[main] 使用设备: cuda (或 cpu)
[main] 构建 GraphSAGERecommender 模型 ...
```

**模型结构**:
- `self.emb`: 嵌入层，为所有节点（用户+物品）生成初始嵌入
- `self.layers`: GraphSAGE 卷积层列表

---

### 2.8 初始化优化器

**代码位置**: ```90:91:main_baseline.py```

**处理过程**:
- 创建 Adam 优化器，学习率为 `lr = 1e-3`

---

### 2.9 准备边索引

**代码位置**: ```93:94:main_baseline.py```

**处理过程**:
- 将 `edge_list` 转置为 `[2, E]` 格式（PyTorch Geometric 标准格式）
- 转换为 torch.Tensor 并移动到设备

**格式说明**:
- `edge_list`: `[E, 2]`，每行 `(u, v)`
- `edge_index`: `[2, E]`，第一行是源节点，第二行是目标节点

---

### 2.10 训练循环

**代码位置**: ```96:105:main_baseline.py```

**函数**: ```13:71:core/train/train_baseline.py```

#### 2.10.1 每个 Epoch 的流程

**代码位置**: ```34:71:core/train/train_baseline.py```

**步骤 1: 全图前向传播**

**代码位置**: ```37:44:core/train/train_baseline.py```

**处理过程**:
1. 设置模型为评估模式 (`model.eval()`)
2. 关闭梯度计算 (`torch.no_grad()`)
3. 调用 `model.forward_full(edge_index)`:
   - 获取所有节点的初始嵌入 `self.emb(nodes)`
   - 逐层执行 GraphSAGE 卷积
   - 返回所有节点的嵌入 `all_emb`，形状 `[num_nodes, hidden_dim]`

**代码位置**: ```53:75:core/models/graphsage.py```

**输出信息**:
```
[train] ===== Epoch 0/3 =====
[train] 全图前向用时 <时间>s
```

**中间输出**:
- `all_emb`: `[num_nodes, hidden_dim]` 形状的张量，包含所有节点的嵌入向量

---

**步骤 2: 批次损失计算**

**代码位置**: ```46:68:core/train/train_baseline.py```

**处理过程**:
对每个 batch `(nodes_list, maps_list, triples)`:

1. 遍历 batch 中的每个三元组 `(u, pos, neg)`:
   - 从 `all_emb` 中索引对应的嵌入:
     - `u_emb = all_emb[u]`
     - `pos_emb = all_emb[pos]`
     - `neg_emb = all_emb[neg]`
   
2. 计算评分:
   - `pos_score = model.score(u_emb, pos_emb)` (内积)
   - `neg_score = model.score(u_emb, neg_emb)` (内积)
   
3. 计算损失:
   - 使用 `BCEWithLogitsLoss`
   - `scores = [pos_score, neg_score]`
   - `labels = [1.0, 0.0]`
   - `loss = BCEWithLogitsLoss(scores, labels)`

**注意**: baseline 版本**不执行反向传播**，仅用于验证数据管道和模型前向是否正常。

**输出信息**:
```
Epoch 0: 100%|████████████| <批次总数>/<批次总数> [<时间>s, <速度>it/s]
[train] Epoch 0 平均"伪损失" = <损失值>
```

---

## 完整数据流总结

### 输入文件
1. `data/raw/ratings.dat`: 原始 MovieLens-10M 数据

### 中间文件（预处理阶段生成）
1. `data/processed/train.parquet`: 训练集（uid, iid, rating, timestamp）
2. `data/processed/val.parquet`: 验证集
3. `data/processed/test.parquet`: 测试集
4. `data/graph/edge_list.npy`: 图边列表 `[E, 2]`
5. `data/graph/num_info.npy`: 用户数和物品数统计

### 训练阶段数据流
```
原始数据 (ratings.dat)
  ↓
ID重映射
  ↓
数据集划分 (train/val/test.parquet)
  ↓
构建图结构 (edge_list.npy, num_info.npy)
  ↓
加载图结构 → NeighborSampler
  ↓
加载训练集 → InteractionDataset
  ↓
DataLoader (批次采样 + 邻居采样)
  ↓
GraphSAGERecommender 模型
  ↓
全图前向传播 → all_emb [num_nodes, hidden_dim]
  ↓
批次损失计算（不反向传播）
```

### 关键代码文件索引

| 阶段 | 文件 | 主要函数/类 |
|------|------|------------|
| 数据预处理 | `core/loader/build_graph.py` | `build_user_item_graph()`, `remap_ids_full()`, `temporal_split()` |
| 图加载 | `core/loader/utils_io.py` | `load_graph()` |
| 数据集 | `core/loader/datasets.py` | `InteractionDataset` |
| 采样器 | `core/sampler/neighbor_sampler.py` | `NeighborSampler` |
| 批处理 | `core/sampler/collate_fn.py` | `collate_subgraphs()` |
| 模型 | `core/models/graphsage.py` | `GraphSAGERecommender` |
| 训练 | `core/train/train_baseline.py` | `train_baseline()` |
| 配置 | `core/config.py` | `Config` |
| 入口脚本 | `scripts/run_build_graph.py` | - |
| 训练入口 | `main_baseline.py` | `main()` |

---

## 执行顺序

1. **第一步**: 运行数据预处理
   ```bash
   python scripts/run_build_graph.py
   ```

2. **第二步**: 运行 baseline 训练
   ```bash
   python main_baseline.py
   ```

---

## 注意事项

1. **Baseline 训练特点**: 
   - 每个 epoch 开始时进行一次全图前向传播
   - 训练时使用预计算的嵌入，**不执行反向传播**
   - 仅用于验证数据管道和模型前向是否正常工作

2. **内存使用**: 
   - 数据预处理使用分块读取，避免内存溢出
   - 训练时全图嵌入会占用较多内存（`num_nodes × hidden_dim`）

3. **采样策略**: 
   - 负采样采用均匀随机采样（不排除正样本）
   - 邻居采样采用 GraphSAGE 风格的多层采样

