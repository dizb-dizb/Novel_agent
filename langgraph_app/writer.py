# -*- coding: utf-8 -*-
"""
LangGraph Studio 入口：暴露写作图供官方 UI 可视化与调试。
运行 langgraph dev 后，在 Studio 中可选择 writer 图，查看节点、状态与执行轨迹。
"""
import sys
from pathlib import Path

# 项目根加入 path，保证可 import src
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.writer.graph_workflow import build_writer_graph

# Studio 要求：名为 graph 的变量，为已 compile 的图
graph = build_writer_graph()
