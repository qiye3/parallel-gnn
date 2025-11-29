# 快速开始指南 (Quick Start)

欢迎使用并行图神经网络推荐系统！本指南将帮助你在几分钟内运行第一个训练示例。

---

## 📋 目录

- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [主要功能](#主要功能)
- [常见问题](#常见问题)

---

## 🔧 环境要求

### 必需环境

- **Python**: 3.8 或更高版本
- **PyTorch**: 1.10.0 或更高版本（支持 CUDA 更佳）
- **NumPy**: 1.20.0 或更高版本
- **Pandas**: 1.3.0 或更高版本

### 可选环境（用于并行训练）

- **CUDA**: 11.0 或更高版本（多 GPU 训练需要）
- **NCCL**: 多 GPU 通信后端（通常随 PyTorch 安装）

---

## 📦 安装步骤

### 1. 克隆或下载项目

```bash
# 如果使用 git
git clone <repository-url>
cd parallel-gnn

# 或直接解压项目压缩包
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n parallel-gnn python=3.8
conda activate parallel-gnn

# 或使用 venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. 安装依赖

```bash
# 安装 PyTorch（根据你的 CUDA 版本选择，或使用 CPU 版本）
# CUDA 11.8 示例
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 CPU 版本
pip install torch torchvision torchaudio

# 安装其他依赖
pip install numpy pandas tqdm pyarrow
```

### 4. 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 📊 数据准备

### 1. 准备原始数据

将 MovieLens-32M 数据集放置在 `data/raw/` 目录下：

```
data/raw/
  ├── ratings.csv    # 必需：用户评分数据
  ├── movies.csv     # 可选：电影元数据
  ├── tags.csv       # 可选：标签数据
  └── links.csv      # 可选：链接数据
```

**注意**: 如果 `data/raw/ratings.csv` 已存在，可以跳过此步骤。

### 2. 构建图结构

运行图构建脚本，将原始数据转换为图结构和训练/验证/测试集：

```bash
python scripts/run_build_graph.py
```

该脚本会执行以下操作：
- 读取 `data/raw/ratings.csv`
- 重新映射用户和物品 ID
- 按时间戳划分训练/验证/测试集
- 构建用户-物品二部图
- 保存处理后的数据到 `data/processed/` 和 `data/graph/`

**预期输出**:
```
=== Building MovieLens-32M graph ===
=== Pass 1: scanning all IDs ===
  Users: 200948
  Items: 84432
=== Pass 2: remapping IDs & concatenating ===
  Total interactions: 25600164
=== Splitting train/val/test ===
=== Building edge list ===
=== Done ===
```

**处理时间**: 根据机器性能，通常需要 5-15 分钟。

---

## 🚀 快速开始

### 方式 1: 快速训练（推荐新手）

运行快速训练脚本，使用少量样本验证系统是否正常工作：

```bash
python scripts/quick_train.py
```

该脚本会：
- 从训练集中随机采样 1,000 条样本
- 运行 1 个 epoch
- 输出训练日志和损失信息

**预期输出**:
```
[quick_train] 加载图结构 ...
[quick_train] 图加载完成: 边数 = 25600164, 用户数 = 200948, 物品数 = 84432
[quick_train] 构建 NeighborSampler, num_neighbors = [15, 10]
...
[train] ===== Epoch 0/0 =====
[train] 全图前向用时 3.24s
Epoch 0: 100%|████████████| 1/1 [00:05<00:00, 5.23s/it]
[train] Epoch 0 平均"伪损失" = 0.6931
```

### 方式 2: 完整基线训练

运行完整的基线训练（使用全量训练数据）：

```bash
python main_baseline.py
```

或使用脚本：

```bash
bash scripts/run_baseline.sh
```

**注意**: 完整训练会使用所有训练数据，可能需要较长时间。

### 方式 3: Mini-batch 训练

运行真正的 mini-batch GraphSAGE 训练（每个 batch 构建子图）：

```bash
python scripts/train_mini_batch_demo.py
```

---

## 🎯 主要功能

### 1. 基线训练 (`main_baseline.py`)

**特点**:
- 每个 epoch 先做一次全图前向传播
- 基于预计算的节点嵌入计算损失
- 适合快速验证和实验

**使用场景**: 快速实验、模型验证、小规模数据

### 2. Mini-batch 训练 (`scripts/train_mini_batch_demo.py`)

**特点**:
- 每个 batch 动态构建子图
- 在子图上执行 GNN 前向和反向传播
- 显存占用更可控

**使用场景**: 大规模图训练、显存受限环境

### 3. 多进程数据加载 (`parallel/dataloader_mp.py`)

**特点**:
- 使用多个 worker 进程并行采样
- 加速数据准备阶段
- 减少主进程负担

**使用场景**: CPU 采样成为瓶颈时

### 4. 分布式训练 (`parallel/train_ddp.py`)

**特点**:
- 使用 PyTorch DDP 实现多 GPU 训练
- 自动同步梯度
- 支持单机多卡和多机多卡

**使用方式**:
```bash
# 2 GPU 训练
torchrun --nproc_per_node=2 -m parallel.train_ddp

# 4 GPU 训练
torchrun --nproc_per_node=4 -m parallel.train_ddp
```

---

## ⚙️ 配置参数

主要配置参数在 `core/config.py` 中定义：

```python
class Config:
    # 模型参数
    hidden_dim = 64          # 隐藏层维度
    num_layers = 2           # GNN 层数
    num_neighbors = [15, 10] # 每层采样邻居数

    # 训练参数
    lr = 1e-3                # 学习率
    batch_size = 1024        # 批次大小
    epochs = 4               # 训练轮数

    # 评估参数
    K = 10                   # Top-K 推荐
    num_neg = 100            # 负采样数量
```

可以根据需要修改这些参数。

---

## 📁 项目结构

```
parallel-gnn/
├── core/                    # 核心模块
│   ├── loader/              # 数据加载与图构建
│   ├── models/              # 模型定义（GraphSAGE / GAT）
│   ├── sampler/             # 子图采样
│   └── train/               # 训练与评估
├── parallel/                 # 并行化模块
│   ├── dataloader_mp.py     # 多进程数据加载
│   ├── train_ddp.py         # DDP 分布式训练
│   └── dist_inference.py    # 分布式推理
├── scripts/                 # 运行脚本
│   ├── run_build_graph.py   # 构建图
│   ├── quick_train.py       # 快速训练
│   └── train_mini_batch_demo.py  # Mini-batch 训练
├── data/                     # 数据目录
│   ├── raw/                  # 原始数据
│   ├── processed/            # 处理后数据
│   └── graph/                # 图结构文件
├── main_baseline.py          # 基线训练入口
├── QUICKSTART.md             # 本文件
├── PROJECT_FLOW.md           # 项目流程文档
└── EXECUTION_FLOW.md         # 执行流程文档
```

---

## ❓ 常见问题

### Q1: 运行 `run_build_graph.py` 时提示 "No module named 'core'"

**解决方案**: 确保在项目根目录下运行脚本：

```bash
# 正确：在项目根目录
cd parallel-gnn
python scripts/run_build_graph.py

# 错误：在 scripts 目录
cd scripts
python run_build_graph.py  # 会报错
```

### Q2: 训练时显存不足 (OOM)

**解决方案**:
1. 减小 `batch_size`（在 `core/config.py` 中修改）
2. 减小 `num_neighbors`（如改为 `[10, 5]`）
3. 使用 `train_mini_batch_demo.py` 而不是 `main_baseline.py`
4. 使用更少的训练样本（在 `quick_train.py` 中调整 `SAMPLE_SIZE`）

### Q3: 训练速度很慢

**可能原因和解决方案**:
1. **CPU 采样瓶颈**: 使用多进程数据加载（`parallel/dataloader_mp.py`）
2. **单 GPU 训练**: 使用 DDP 多 GPU 训练（`parallel/train_ddp.py`）
3. **数据加载慢**: 增加 `num_workers`（在 DataLoader 中）

### Q4: CUDA 相关错误

**解决方案**:
1. 检查 CUDA 版本是否与 PyTorch 匹配：
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```
2. 如果只有 CPU，确保使用 CPU 版本的 PyTorch
3. 在代码中强制使用 CPU：`device = "cpu"`

### Q5: 如何查看训练进度？

训练过程中会输出：
- 每个 epoch 的开始和结束信息
- 全图前向传播耗时
- 每个 batch 的损失（每 100 个 batch 打印一次）
- Epoch 平均损失

### Q6: 如何评估模型性能？

使用 `core/train/eval.py` 中的 `evaluate` 函数：

```python
from core.train.eval import evaluate
import pandas as pd

# 加载测试集
test_df = pd.read_parquet("data/processed/test.parquet")

# 计算全图嵌入
with torch.no_grad():
    all_emb = model.forward_full(edge_index)

# 评估
hr, ndcg = evaluate(model, all_emb, test_df, K=10, num_neg=100)
print(f"HR@10: {hr:.4f}, NDCG@10: {ndcg:.4f}")
```

---

## 📚 更多文档

- **项目流程**: 查看 `PROJECT_FLOW.md` 了解完整的项目流程
- **执行流程**: 查看 `EXECUTION_FLOW.md` 了解详细的代码执行路径
- **实验报告**: 查看 `REPORT.md` 了解实验设计和结果

---

## 🎓 下一步

1. ✅ 完成快速训练，验证系统正常工作
2. 📊 尝试修改配置参数，观察训练效果
3. 🔬 运行不同的训练模式（baseline / mini-batch / DDP）
4. 📈 使用评估函数测试模型性能
5. 🚀 探索并行化优化策略

---

## 💡 提示

- **首次运行**: 建议先运行 `quick_train.py` 验证环境配置
- **调试模式**: 使用小数据集和少量 epoch 快速迭代
- **生产训练**: 使用完整数据集和 DDP 多 GPU 训练
- **性能优化**: 根据瓶颈选择相应的并行化策略

---

**祝你使用愉快！如有问题，请查看项目文档或提交 Issue。**

