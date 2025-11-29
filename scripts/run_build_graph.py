"""
脚本入口：构建 MovieLens 用户-物品二部图及训练/验证/测试集。

推荐使用方式:
    python scripts/run_build_graph.py
"""

import os
import sys

# ---------- 确保可以导入项目根目录下的 core 包 ----------
# 本脚本位于 parallel-gnn/scripts/ 目录下，
# 需要手动把项目根目录加入 sys.path，才能 import core.*
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.loader.build_graph import build_user_item_graph


if __name__ == "__main__":
    build_user_item_graph()
