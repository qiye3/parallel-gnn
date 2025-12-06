"""
run_all_experiments.py
----------------------
完整的并行实验脚本，自动运行四类实验并记录结果到logs目录。

实验1：基线（串行）- 单线程、单GPU
实验2：多进程DataLoader - 测试不同num_workers
实验3：分布式训练（DDP）- 1 GPU vs 2 GPU
实验4：分布式推理 - 单GPU vs 多GPU全图前向

所有结果写入 logs/ 目录
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.distributed as dist
from datetime import datetime
from pathlib import Path

# 确保可以导入项目包
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config import Config
from core.loader.utils_io import load_graph
from core.loader.datasets import InteractionDataset
from core.sampler.neighbor_sampler import NeighborSampler
from core.sampler.collate_fn import collate_subgraphs
from parallel.dataloader_mp import create_mp_dataloader
from core.models.graphsage import GraphSAGERecommender
from core.train.train_baseline import train_baseline


# 创建logs目录
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


def log_to_file(filepath, content, mode="a"):
    """将内容追加到日志文件"""
    with open(filepath, mode, encoding="utf-8") as f:
        f.write(content + "\n")
    print(content)


def save_results(exp_name, results):
    """保存实验结果到JSON文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = LOGS_DIR / f"{exp_name}_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return json_path


# ============================================================================
# 实验1：基线（串行）- 单线程、单GPU
# ============================================================================

def experiment1_baseline():
    """实验1：基线串行训练"""
    print("\n" + "="*80)
    print("实验1：基线（串行）- 单线程DataLoader、单进程采样、单GPU训练")
    print("="*80)
    
    log_file = LOGS_DIR / "exp1_baseline.log"
    log_to_file(log_file, f"\n{'='*80}", mode="w")
    log_to_file(log_file, f"实验1：基线（串行）")
    log_to_file(log_file, f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    cfg = Config()
    
    # 加载图结构
    print("[Exp1] 加载图结构...")
    edge_list, num_users, num_items = load_graph()
    log_to_file(log_file, f"图信息: 边数={edge_list.shape[0]}, 用户数={num_users}, 物品数={num_items}")
    
    # 构建单进程DataLoader
    print("[Exp1] 构建单进程DataLoader...")
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
        num_workers=0,  # 单进程
        collate_fn=_collate,
    )
    
    # 模型和优化器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Exp1] 使用设备: {device}")
    log_to_file(log_file, f"设备: {device}")
    
    model = GraphSAGERecommender(
        num_users, num_items,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    edge_index = torch.tensor(edge_list.T, dtype=torch.long).to(device)
    
    # 训练并记录指标
    print("[Exp1] 开始训练...")
    try:
        metrics = train_baseline(model, loader, edge_index, optimizer, device=device, epochs=cfg.epochs, record_metrics=True)
    except Exception as e:
        log_to_file(log_file, f"训练失败: {e}")
        print(f"[Exp1] 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    
    # 记录结果
    results = {
        "experiment": "基线（串行）",
        "config": {
            "batch_size": cfg.batch_size,
            "num_workers": 0,
            "num_gpus": 1,
            "epochs": cfg.epochs,
        },
        "metrics": {
            "epoch_times": metrics["epoch_times"],
            "forward_times": metrics["forward_times"],
            "sampling_times": metrics["sampling_times"],
            "avg_losses": metrics["avg_losses"],
            "avg_epoch_time": np.mean(metrics["epoch_times"]),
            "avg_forward_time": np.mean(metrics["forward_times"]),
            "avg_sampling_time": np.mean(metrics["sampling_times"]),
        }
    }
    
    json_path = save_results("exp1_baseline", results)
    log_to_file(log_file, f"\n结果已保存到: {json_path}")
    log_to_file(log_file, f"平均epoch时间: {results['metrics']['avg_epoch_time']:.2f}s")
    log_to_file(log_file, f"平均前向时间: {results['metrics']['avg_forward_time']:.2f}s")
    log_to_file(log_file, f"平均采样时间: {results['metrics']['avg_sampling_time']:.2f}s")
    
    return results


# ============================================================================
# 实验2：多进程DataLoader
# ============================================================================

def experiment2_mp_dataloader():
    """实验2：多进程DataLoader性能对比"""
    print("\n" + "="*80)
    print("实验2：多进程DataLoader - 测试 num_workers ∈ {0, 2, 4, 8}")
    print("="*80)
    
    log_file = LOGS_DIR / "exp2_mp_dataloader.log"
    log_to_file(log_file, f"\n{'='*80}", mode="w")
    log_to_file(log_file, f"实验2：多进程DataLoader")
    log_to_file(log_file, f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    cfg = Config()
    num_workers_list = [0, 2, 4, 8]
    all_results = {}
    
    # 加载图结构
    print("[Exp2] 加载图结构...")
    edge_list, num_users, num_items = load_graph()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for num_workers in num_workers_list:
        print(f"\n[Exp2] 测试 num_workers = {num_workers}")
        log_to_file(log_file, f"\n--- num_workers = {num_workers} ---")
        
        # 构建DataLoader
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
        
        # 模型和优化器
        model = GraphSAGERecommender(
            num_users, num_items,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        edge_index = torch.tensor(edge_list.T, dtype=torch.long).to(device)
        
        try:
            # 训练并记录指标
            metrics = train_baseline(model, loader, edge_index, optimizer, device=device, epochs=cfg.epochs, record_metrics=True)
            
            # 记录结果
            results = {
                "num_workers": num_workers,
                "metrics": {
                    "epoch_times": metrics["epoch_times"],
                    "forward_times": metrics["forward_times"],
                    "sampling_times": metrics["sampling_times"],
                    "avg_losses": metrics["avg_losses"],
                    "avg_epoch_time": np.mean(metrics["epoch_times"]),
                    "avg_forward_time": np.mean(metrics["forward_times"]),
                    "avg_sampling_time": np.mean(metrics["sampling_times"]),
                }
            }
            
            all_results[f"num_workers_{num_workers}"] = results
            
            log_to_file(log_file, f"平均epoch时间: {results['metrics']['avg_epoch_time']:.2f}s")
            log_to_file(log_file, f"平均前向时间: {results['metrics']['avg_forward_time']:.2f}s")
            log_to_file(log_file, f"平均采样时间: {results['metrics']['avg_sampling_time']:.2f}s")
        except Exception as e:
            log_to_file(log_file, f"num_workers={num_workers} 测试失败: {e}")
            print(f"[Exp2] num_workers={num_workers} 测试失败: {e}")
            all_results[f"num_workers_{num_workers}"] = {"error": str(e)}
        finally:
            # 清理GPU内存
            del model, optimizer, edge_index
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 计算加速比（相对于num_workers=0）
    speedups = {}
    if "num_workers_0" in all_results and "metrics" in all_results["num_workers_0"]:
        baseline_time = all_results["num_workers_0"]["metrics"]["avg_epoch_time"]
        for num_workers in num_workers_list:
            if num_workers > 0:
                worker_key = f"num_workers_{num_workers}"
                if worker_key in all_results and "metrics" in all_results[worker_key]:
                    worker_time = all_results[worker_key]["metrics"]["avg_epoch_time"]
                    speedup = baseline_time / worker_time
                    speedups[num_workers] = speedup
                    log_to_file(log_file, f"num_workers={num_workers} 加速比: {speedup:.2f}x")
    
    # 保存结果
    final_results = {
        "experiment": "多进程DataLoader",
        "config": {
            "batch_size": cfg.batch_size,
            "num_workers_list": num_workers_list,
            "epochs": cfg.epochs,
        },
        "results": all_results,
        "speedups": speedups,
    }
    
    json_path = save_results("exp2_mp_dataloader", final_results)
    log_to_file(log_file, f"\n结果已保存到: {json_path}")
    
    return final_results


# ============================================================================
# 实验3：分布式训练（DDP）
# ============================================================================

def experiment3_ddp():
    """实验3：分布式训练（DDP）- 1 GPU vs 2 GPU"""
    print("\n" + "="*80)
    print("实验3：分布式训练（DDP）- 1 GPU vs 2 GPU")
    print("="*80)
    print("注意：此实验需要手动运行，因为需要torchrun启动多个进程")
    print("请分别运行以下命令：")
    print("  1 GPU: python main_baseline.py")
    print("  2 GPU: torchrun --nproc_per_node=2 -m parallel.train_ddp")
    print("="*80)
    
    log_file = LOGS_DIR / "exp3_ddp.log"
    log_to_file(log_file, f"\n{'='*80}", mode="w")
    log_to_file(log_file, f"实验3：分布式训练（DDP）")
    log_to_file(log_file, f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_to_file(log_file, "\n注意：此实验需要手动运行torchrun命令")
    
    return {"note": "需要手动运行torchrun命令"}


# ============================================================================
# 实验4：分布式推理
# ============================================================================

def experiment4_dist_inference():
    """实验4：分布式推理 - 单GPU vs 多GPU全图前向"""
    print("\n" + "="*80)
    print("实验4：分布式推理 - 单GPU vs 多GPU全图前向")
    print("="*80)
    
    log_file = LOGS_DIR / "exp4_dist_inference.log"
    log_to_file(log_file, f"\n{'='*80}", mode="w")
    log_to_file(log_file, f"实验4：分布式推理")
    log_to_file(log_file, f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    cfg = Config()
    
    # 加载图结构
    print("[Exp4] 加载图结构...")
    edge_list, num_users, num_items = load_graph()
    log_to_file(log_file, f"图信息: 边数={edge_list.shape[0]}, 用户数={num_users}, 物品数={num_items}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not torch.cuda.is_available():
        print("[Exp4] 警告：未检测到GPU，无法进行多GPU推理实验")
        log_to_file(log_file, "警告：未检测到GPU，无法进行多GPU推理实验")
        return {"note": "需要GPU支持"}
    
    # 单GPU推理
    print("[Exp4] 测试单GPU推理...")
    model = GraphSAGERecommender(
        num_users, num_items,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    ).to(device)
    
    edge_index = torch.tensor(edge_list.T, dtype=torch.long).to(device)
    
    # 预热
    with torch.no_grad():
        _ = model.forward_full(edge_index)
    
    # 测试单GPU推理时间
    single_gpu_times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model.forward_full(edge_index)
        torch.cuda.synchronize()
        single_gpu_times.append(time.time() - start)
    
    single_gpu_avg = np.mean(single_gpu_times)
    log_to_file(log_file, f"单GPU推理平均时间: {single_gpu_avg:.4f}s")
    
    # 多GPU推理（需要手动运行torchrun）
    print("[Exp4] 多GPU推理需要手动运行torchrun命令")
    log_to_file(log_file, "\n多GPU推理需要手动运行以下命令：")
    log_to_file(log_file, "  torchrun --nproc_per_node=2 -m parallel.train_ddp_inference")
    
    results = {
        "experiment": "分布式推理",
        "single_gpu": {
            "times": single_gpu_times,
            "avg_time": float(single_gpu_avg),
        },
        "note": "多GPU推理需要手动运行torchrun命令",
    }
    
    json_path = save_results("exp4_dist_inference", results)
    log_to_file(log_file, f"\n结果已保存到: {json_path}")
    
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有实验"""
    print("\n" + "="*80)
    print("并行图神经网络推荐系统 - 完整实验流程")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志目录: {LOGS_DIR.absolute()}")
    print("="*80)
    
    # 检查数据文件是否存在
    required_files = [
        "data/processed/train.parquet",
        "data/graph/edge_list.npy",
        "data/graph/num_info.npy",
    ]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"\n错误：缺少必要的数据文件：{missing_files}")
        print("请先运行: python scripts/run_build_graph.py")
        return
    
    all_results = {}
    
    try:
        # 实验1：基线
        print("\n开始实验1...")
        all_results["exp1"] = experiment1_baseline()
        
        # 实验2：多进程DataLoader
        print("\n开始实验2...")
        all_results["exp2"] = experiment2_mp_dataloader()
        
        # 实验3：DDP（需要手动运行）
        print("\n开始实验3...")
        all_results["exp3"] = experiment3_ddp()
        
        # 实验4：分布式推理
        print("\n开始实验4...")
        all_results["exp4"] = experiment4_dist_inference()
        
    except Exception as e:
        print(f"\n实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        log_file = LOGS_DIR / "error.log"
        log_to_file(log_file, f"错误: {e}\n{traceback.format_exc()}")
    
    # 保存汇总结果
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": all_results,
    }
    summary_path = save_results("all_experiments_summary", summary)
    
    print("\n" + "="*80)
    print("所有实验完成！")
    print(f"汇总结果已保存到: {summary_path}")
    print(f"详细日志请查看: {LOGS_DIR.absolute()}")
    print("="*80)


if __name__ == "__main__":
    main()