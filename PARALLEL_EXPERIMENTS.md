# 并行实验运行指南

本文档详细说明如何运行各种并行实验，包括多进程数据加载和分布式数据并行（DDP）训练。

---

## 前置准备

### 1. 确保数据已预处理

在运行任何并行实验之前，必须先完成数据预处理：

```bash
python scripts/run_build_graph.py
```

这会生成：
- `data/processed/train.parquet`
- `data/graph/edge_list.npy`
- `data/graph/num_info.npy`

---

## 并行实验类型

### 实验1: 多进程数据加载（单GPU）

**特点**:
- 使用多个 worker 进程并行进行邻居采样
- 加速数据准备阶段
- 适合 CPU 采样成为瓶颈的场景

**代码位置**: `parallel/dataloader_mp.py`

**实现原理**:
- 使用 PyTorch DataLoader 的 `num_workers > 0` 机制
- 每个 worker 进程维护独立的 `NeighborSampler` 实例
- 采样逻辑在 worker 进程中并行执行

**使用方法**:

创建一个使用多进程数据加载的训练脚本，例如：

```python
# 示例：使用多进程数据加载的训练脚本
from parallel.dataloader_mp import create_mp_dataloader
from core.loader.utils_io import load_graph
from core.config import Config
from core.models.graphsage import GraphSAGERecommender
from core.train.train_baseline import train_baseline
import torch

cfg = Config()
edge_list, num_users, num_items = load_graph()

# 使用多进程数据加载器（4个worker）
loader = create_mp_dataloader(
    parquet_path="data/processed/train.parquet",
    num_items=num_items,
    edge_list=edge_list,
    num_neighbors=cfg.num_neighbors,
    batch_size=cfg.batch_size,
    num_workers=4,  # 使用4个worker进程
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GraphSAGERecommender(
    num_users, num_items,
    hidden_dim=cfg.hidden_dim,
    num_layers=cfg.num_layers,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
edge_index = torch.tensor(edge_list.T, dtype=torch.long).to(device)

train_baseline(model, loader, edge_index, optimizer, device=device, epochs=cfg.epochs)
```

**参数说明**:
- `num_workers`: worker 进程数量，建议设置为 CPU 核心数的一半到全部
- `batch_size`: 批次大小，可以适当增大以充分利用并行采样

**性能对比实验**:

可以修改 `experiments/exp_mp_sampler.py` 来对比不同 `num_workers` 的性能：

```python
# experiments/exp_mp_sampler.py 示例
import time
from parallel.dataloader_mp import create_mp_dataloader
from core.loader.utils_io import load_graph
from core.config import Config

cfg = Config()
edge_list, num_users, num_items = load_graph()

# 测试不同 num_workers 的性能
for num_workers in [0, 2, 4, 8]:
    loader = create_mp_dataloader(
        parquet_path="data/processed/train.parquet",
        num_items=num_items,
        edge_list=edge_list,
        num_neighbors=cfg.num_neighbors,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
    )
    
    start = time.time()
    for i, batch in enumerate(loader):
        if i >= 10:  # 只测试前10个batch
            break
    elapsed = time.time() - start
    
    print(f"num_workers={num_workers}: {elapsed:.2f}s for 10 batches")
```

---

### 实验2: 分布式数据并行训练（DDP - 多GPU）

**特点**:
- 使用 PyTorch DDP 实现多 GPU 训练
- 自动同步梯度
- 支持单机多卡和多机多卡

**代码位置**: `parallel/train_ddp.py`

**实现原理**:
1. 使用 `torchrun` 启动多个进程，每个进程绑定一个 GPU
2. 使用 `DistributedSampler` 将数据划分给不同进程
3. 使用 `DistributedDataParallel (DDP)` 包装模型，自动同步梯度
4. 每个进程独立前向传播，DDP 自动聚合梯度并同步更新参数

**使用方法**:

#### 方式1: 使用 torchrun 命令（推荐）

**2 GPU 训练**:
```bash
torchrun --nproc_per_node=2 -m parallel.train_ddp
```

**4 GPU 训练**:
```bash
torchrun --nproc_per_node=4 -m parallel.train_ddp
```

**8 GPU 训练**:
```bash
torchrun --nproc_per_node=8 -m parallel.train_ddp
```

#### 方式2: 使用脚本（Windows/Linux）

**2 GPU 训练**:
```bash
# Linux/Mac
bash scripts/run_ddp_2gpu.sh

# Windows (需要先安装 Git Bash 或使用 WSL)
# 或直接运行：
torchrun --nproc_per_node=2 -m parallel.train_ddp
```

**4 GPU 训练**:
```bash
# Linux/Mac
bash scripts/run_ddp_4gpu.sh

# Windows
torchrun --nproc_per_node=4 -m parallel.train_ddp
```

#### 方式3: 自定义 GPU 数量

```bash
# N GPU 训练（N 为你的 GPU 数量）
torchrun --nproc_per_node=N -m parallel.train_ddp
```

**环境要求**:
- 需要至少 2 个 GPU（单 GPU 请使用单进程训练）
- 确保 PyTorch 支持 CUDA
- 确保所有 GPU 可见（使用 `nvidia-smi` 检查）

**训练流程** (```41:106:parallel/train_ddp.py```):

1. **初始化进程组** (```21:38:parallel/train_ddp.py```):
   - 使用 NCCL 后端（GPU）或 Gloo 后端（CPU）
   - 每个进程绑定对应的 GPU（rank 0 → GPU 0, rank 1 → GPU 1, ...）

2. **数据并行化** (```63:83:parallel/train_ddp.py```):
   - 使用 `DistributedSampler` 将数据集划分给不同进程
   - 每个进程只处理自己负责的数据子集

3. **模型并行化** (```88:98:parallel/train_ddp.py```):
   - 使用 `DistributedDataParallel (DDP)` 包装模型
   - DDP 自动处理梯度同步

4. **训练循环** (```103:103:parallel/train_ddp.py```):
   - 调用 `train_baseline()`，此时模型已经是 DDP 封装
   - 每个进程独立前向传播
   - DDP 自动聚合梯度并同步更新参数

**注意事项**:
- 每个进程会输出自己的日志，可能看起来比较混乱
- 建议只让 rank 0 进程输出重要信息（可以在代码中添加 `if rank == 0:` 判断）
- 确保每个 GPU 有足够的显存

---

### 实验3: 多进程采样性能对比

**代码位置**: `experiments/exp_mp_sampler.py`

**目的**: 对比不同 `num_workers` 设置下的数据加载性能

**当前状态**: 占位符，需要实现具体实验逻辑

**建议实现**:

```python
"""
experiments/exp_mp_sampler.py
对比多进程 DataLoader 与单进程 DataLoader 的性能差异
"""
import time
import torch
from parallel.dataloader_mp import create_mp_dataloader
from core.loader.datasets import InteractionDataset
from core.loader.utils_io import load_graph
from core.sampler.neighbor_sampler import NeighborSampler
from core.sampler.collate_fn import collate_subgraphs
from core.config import Config

def test_dataloader_performance(num_workers, num_batches=50):
    """测试指定 num_workers 下的数据加载性能"""
    cfg = Config()
    edge_list, num_users, num_items = load_graph()
    
    if num_workers == 0:
        # 单进程版本
        dataset = InteractionDataset("data/processed/train.parquet", num_items)
        sampler = NeighborSampler(edge_list, cfg.num_neighbors)
        
        def _collate(batch):
            processed = []
            for u, pos, neg in batch:
                nodes, node2idx = sampler.sample([u, pos, neg])
                processed.append(((nodes, node2idx), (u, pos, neg)))
            return collate_subgraphs(processed)
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=_collate,
        )
    else:
        # 多进程版本
        loader = create_mp_dataloader(
            parquet_path="data/processed/train.parquet",
            num_items=num_items,
            edge_list=edge_list,
            num_neighbors=cfg.num_neighbors,
            batch_size=cfg.batch_size,
            num_workers=num_workers,
        )
    
    # 预热
    for i, _ in enumerate(loader):
        if i >= 5:
            break
    
    # 正式测试
    start = time.time()
    total_samples = 0
    for i, (nodes_list, maps_list, triples) in enumerate(loader):
        total_samples += len(triples)
        if i >= num_batches:
            break
    elapsed = time.time() - start
    
    throughput = total_samples / elapsed
    return elapsed, throughput

if __name__ == "__main__":
    print("多进程采样性能对比实验")
    print("=" * 50)
    
    num_batches = 50
    for num_workers in [0, 2, 4, 8]:
        elapsed, throughput = test_dataloader_performance(num_workers, num_batches)
        print(f"num_workers={num_workers:2d}: "
              f"时间={elapsed:.2f}s, "
              f"吞吐量={throughput:.1f} 样本/秒")
```

---

## 实验执行顺序建议

### 1. 快速验证（单进程）

首先确保单进程训练可以正常运行：

```bash
python scripts/quick_train.py
```

或完整训练：

```bash
python main_baseline.py
```

### 2. 多进程数据加载实验

创建一个测试脚本，对比不同 `num_workers` 的性能：

```bash
python experiments/exp_mp_sampler.py
```

### 3. DDP 多GPU训练

确保有多个 GPU 后，运行：

```bash
# 2 GPU
torchrun --nproc_per_node=2 -m parallel.train_ddp

# 4 GPU
torchrun --nproc_per_node=4 -m parallel.train_ddp
```

---

## 性能监控

### 监控 GPU 使用情况

在另一个终端运行：

```bash
# Linux
watch -n 1 nvidia-smi

# 或使用
nvidia-smi -l 1
```

### 监控 CPU 使用情况

```bash
# Linux
htop

# 或
top
```

### 监控进程

```bash
# 查看所有 Python 进程
ps aux | grep python

# 查看 GPU 进程
nvidia-smi
```

---

## 常见问题

### 1. DDP 训练时出现 "Address already in use"

**原因**: 端口被占用

**解决**: 
- 等待之前的训练进程完全结束
- 或设置不同的端口：`export MASTER_PORT=29501`

### 2. 多进程数据加载时出现死锁

**原因**: 
- `num_workers` 设置过大
- 数据加载逻辑中有共享资源冲突

**解决**:
- 减少 `num_workers` 数量
- 确保每个 worker 有独立的采样器实例（已在 `create_mp_dataloader` 中实现）

### 3. GPU 显存不足

**解决**:
- 减小 `batch_size`
- 减少 `num_neighbors`（采样更少的邻居）
- 减少 `hidden_dim` 或 `num_layers`

### 4. Windows 上无法使用多进程 DataLoader

**原因**: Windows 上多进程需要 `if __name__ == "__main__":` 保护

**解决**: 确保主脚本有正确的保护：

```python
if __name__ == "__main__":
    main()
```

---

## 实验记录建议

建议在 `experiments/` 目录下记录实验结果：

- `experiments/exp_mp_sampler.md`: 多进程采样性能对比结果
- `experiments/exp_ddp_training.md`: DDP 训练结果（已有）
- `experiments/exp_baseline.md`: 基线实验结果（已有）

记录内容可以包括：
- 实验配置（GPU数量、batch_size、num_workers等）
- 训练时间
- 吞吐量（样本/秒）
- GPU/CPU 利用率
- 内存使用情况

---

## 总结

项目支持以下并行实验方式：

1. **多进程数据加载** (`parallel/dataloader_mp.py`):
   - 适合单 GPU 场景
   - 加速数据准备阶段
   - 使用 `num_workers > 0` 启用

2. **分布式数据并行训练** (`parallel/train_ddp.py`):
   - 适合多 GPU 场景
   - 使用 `torchrun` 启动
   - 自动同步梯度

3. **性能对比实验** (`experiments/exp_mp_sampler.py`):
   - 对比不同并行策略的性能
   - 需要自行实现实验逻辑

根据你的硬件配置选择合适的并行策略，可以显著提升训练效率！

