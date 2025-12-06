# 代码冗余分析报告

## 发现的冗余代码

### 1. 训练函数冗余 ⚠️ **高优先级**

**问题**: `core/train/train_baseline.py` 和 `core/train/train_with_metrics.py` 几乎完全相同

**冗余度**: ~90% 代码重复

**差异**:
- `train_baseline`: 简单版本，只打印基本日志
- `train_with_metrics`: 增加了详细的性能指标记录（epoch时间、前向时间、采样时间等）

**建议**:
- **方案1（推荐）**: 合并为一个函数，通过参数控制是否记录详细指标
  ```python
  def train_baseline(model, dataloader, edge_index, optimizer, 
                     device="cpu", epochs=1, record_metrics=False):
      if record_metrics:
          # 记录详细指标
      else:
          # 简单版本
  ```
- **方案2**: 让 `train_with_metrics` 继承或调用 `train_baseline`，只添加指标记录部分

---

### 2. 主训练脚本冗余 ⚠️ **中优先级**

**问题**: `main_baseline.py` 和 `main_train_mp.py` 有大量重复代码

**冗余度**: ~85% 代码重复

**差异**:
- `main_baseline.py`: 使用单进程 DataLoader (`num_workers=0`)
- `main_train_mp.py`: 使用多进程 DataLoader (`create_mp_dataloader`)

**重复部分**:
- 配置加载
- 图结构加载
- 模型初始化
- 优化器创建
- 训练循环调用

**建议**:
- 提取公共部分为函数，例如：
  ```python
  def setup_training(cfg, use_mp=False, num_workers=4):
      # 加载图结构
      # 创建 DataLoader（根据 use_mp 参数选择）
      # 创建模型和优化器
      return model, loader, edge_index, optimizer, device
  ```

---

### 3. 快速训练脚本冗余 ⚠️ **中优先级**

**问题**: `scripts/quick_train.py` 和 `scripts/quick_train_mp.py` 有大量重复代码

**冗余度**: ~80% 代码重复

**差异**:
- `quick_train.py`: 单进程 + 数据子集采样
- `quick_train_mp.py`: 多进程 + 完整数据集

**重复部分**:
- 图结构加载
- 模型初始化
- 训练循环调用

**建议**:
- 合并为一个脚本，通过参数控制：
  ```python
  def main(use_mp=False, num_workers=4, sample_size=None):
      # 根据参数选择不同的 DataLoader 创建方式
  ```

---

### 4. 占位符文件 ⚠️ **低优先级**

**问题**: 以下文件只有占位符代码，没有实际功能

1. `experiments/exp_mp_sampler.py` - 只有打印语句
2. `parallel/profiler.py` - 只有打印语句

**建议**:
- 如果暂时不需要，可以删除或添加 TODO 注释
- 如果需要保留作为模板，可以添加更详细的实现指南

---

### 5. 未使用的导入 ⚠️ **低优先级**

**问题**: 
- `main_train_mp.py`: 导入了 `numpy` 但未使用
- `main_baseline.py`: 导入了 `numpy` 但未使用

**建议**: 移除未使用的导入

---

## 重构建议优先级

### 高优先级（建议立即处理）
1. ✅ **合并训练函数**: `train_baseline.py` 和 `train_with_metrics.py`
   - 影响范围：所有使用这些函数的脚本
   - 收益：减少维护成本，避免功能不一致

### 中优先级（建议后续处理）
2. ⚠️ **重构主训练脚本**: 提取公共代码
   - 影响范围：`main_baseline.py`, `main_train_mp.py`
   - 收益：减少代码重复，提高可维护性

3. ⚠️ **合并快速训练脚本**: 统一接口
   - 影响范围：`scripts/quick_train.py`, `scripts/quick_train_mp.py`
   - 收益：简化使用，减少维护

### 低优先级（可选）
4. 📝 **清理占位符文件**: 删除或完善
5. 🧹 **清理未使用导入**: 代码整洁

---

## 具体重构方案示例

### 方案1: 合并训练函数

```python
# core/train/train_baseline.py
def train_baseline(model, dataloader, edge_index, optimizer, 
                   device="cpu", epochs=1, record_metrics=False):
    """训练函数，可选择是否记录详细指标"""
    metrics = {} if record_metrics else None
    
    if record_metrics:
        metrics = {
            "epoch_times": [],
            "forward_times": [],
            "sampling_times": [],
            "avg_losses": [],
        }
    
    # ... 训练逻辑 ...
    # 根据 record_metrics 决定是否记录指标
    
    return metrics if record_metrics else None
```

### 方案2: 提取公共训练设置

```python
# core/train/setup.py
def create_dataloader(parquet_path, num_items, edge_list, cfg, 
                     use_mp=False, num_workers=4):
    """统一的 DataLoader 创建函数"""
    if use_mp:
        return create_mp_dataloader(...)
    else:
        # 单进程版本
        return torch.utils.data.DataLoader(...)

def setup_training(cfg, use_mp=False, num_workers=4):
    """统一的训练设置函数"""
    edge_list, num_users, num_items = load_graph()
    loader = create_dataloader(..., use_mp=use_mp, num_workers=num_workers)
    model = GraphSAGERecommender(...)
    optimizer = torch.optim.Adam(...)
    # ...
    return model, loader, edge_index, optimizer, device
```

---

## 总结

**冗余代码统计**:
- 训练函数: 2个文件，~90% 重复
- 主训练脚本: 2个文件，~85% 重复  
- 快速训练脚本: 2个文件，~80% 重复
- 占位符文件: 2个文件，无实际代码

**建议行动**:
1. 立即合并训练函数（高优先级）
2. 后续重构主训练脚本（中优先级）
3. 可选清理占位符和未使用导入（低优先级）

**收益**:
- 减少代码量：预计可减少 ~300-400 行重复代码
- 提高可维护性：修改一处即可影响所有使用场景
- 降低出错风险：避免功能不一致

