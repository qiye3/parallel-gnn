# 并行图神经网络推荐系统

基于 GraphSAGE 的大规模推荐系统并行训练与推理框架，支持多进程采样、分布式训练和全图推理加速。

---

## ✨ 特性

- 🚀 **多种训练模式**: 基线训练、Mini-batch 训练、分布式训练
- ⚡ **并行优化**: 多进程数据加载、DDP 分布式训练、分布式推理
- 📊 **大规模数据支持**: 支持 MovieLens-32M 等大规模数据集
- 🔧 **易于扩展**: 模块化设计，支持自定义模型和采样策略
- 📝 **完整文档**: 详细的代码注释和流程文档

---

## 🚀 快速开始

**新用户请先查看 [QUICKSTART.md](QUICKSTART.md) 获取详细的安装和使用指南！**

### 快速体验（3 步）

```bash
# 1. 安装依赖
pip install torch numpy pandas tqdm pyarrow

# 2. 构建图结构（如果数据已准备好）
python scripts/run_build_graph.py

# 3. 运行快速训练
python scripts/quick_train.py
```

---

## 📁 项目结构

```
parallel-gnn/
├── core/              # 核心模块（数据、模型、训练）
├── parallel/          # 并行化模块（多进程、DDP）
├── scripts/           # 运行脚本
├── data/              # 数据目录
├── main_baseline.py   # 基线训练入口
└── QUICKSTART.md      # 快速开始指南 ⭐
```

---

## 📚 文档

- **[QUICKSTART.md](QUICKSTART.md)** - 快速开始指南（推荐新用户阅读）
- **[PROJECT_FLOW.md](PROJECT_FLOW.md)** - 项目完整流程文档
- **[EXECUTION_FLOW.md](EXECUTION_FLOW.md)** - 代码执行流程详解
- **[REPORT.md](REPORT.md)** - 实验报告与技术文档

---

## 🎯 主要功能

### 1. 基线训练
```bash
python main_baseline.py
```
每个 epoch 先做全图前向，再基于预计算嵌入计算损失。

### 2. Mini-batch 训练
```bash
python scripts/train_mini_batch_demo.py
```
每个 batch 动态构建子图，显存占用更可控。

### 3. 分布式训练（DDP）
```bash
torchrun --nproc_per_node=2 -m parallel.train_ddp
```
使用 PyTorch DDP 实现多 GPU 训练。

### 4. 快速训练（烟雾测试）
```bash
python scripts/quick_train.py
```
使用少量样本快速验证系统是否正常工作。

---

## ⚙️ 配置

主要配置在 `core/config.py` 中：

```python
hidden_dim = 64          # 隐藏层维度
num_layers = 2           # GNN 层数
num_neighbors = [15, 10] # 每层采样邻居数
batch_size = 1024        # 批次大小
lr = 1e-3                # 学习率
epochs = 4               # 训练轮数
```

---

## 🔧 环境要求

- Python 3.8+
- PyTorch 1.10.0+
- NumPy, Pandas, tqdm, pyarrow

详细安装步骤请参考 [QUICKSTART.md](QUICKSTART.md)。

---

## 📖 使用示例

### 数据准备
```bash
# 将 MovieLens-32M 数据放在 data/raw/ 目录下
# 然后运行图构建脚本
python scripts/run_build_graph.py
```

### 训练模型
```bash
# 快速训练（推荐首次使用）
python scripts/quick_train.py

# 完整基线训练
python main_baseline.py

# Mini-batch 训练
python scripts/train_mini_batch_demo.py
```

### 分布式训练
```bash
# 2 GPU 训练
torchrun --nproc_per_node=2 -m parallel.train_ddp

# 4 GPU 训练
torchrun --nproc_per_node=4 -m parallel.train_ddp
```

---

## 🐛 常见问题

**Q: 运行脚本时提示 "No module named 'core'"**  
A: 确保在项目根目录下运行脚本，不要进入子目录。

**Q: 训练时显存不足**  
A: 减小 `batch_size` 或 `num_neighbors`，或使用 `train_mini_batch_demo.py`。

**Q: 训练速度慢**  
A: 使用多进程数据加载或 DDP 多 GPU 训练。

更多问题请查看 [QUICKSTART.md](QUICKSTART.md) 的常见问题部分。

---

## 📄 许可证

本项目用于学术研究和教学目的。

---

## 🙏 致谢

- 基于 GraphSAGE 论文实现
- 使用 PyTorch 和 PyTorch Geometric 相关技术
- 数据集：MovieLens-32M

---

**开始使用**: 查看 [QUICKSTART.md](QUICKSTART.md) 获取详细指南！

