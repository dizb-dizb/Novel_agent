# -*- coding: utf-8 -*-
"""
分层生成流 (Layered Generation Flow)：战略层 → 逻辑层 → 草稿层 → 润色层。
支持 LangGraph 人机循环：Planner → Consultant → Writer → Critique。
"""
from .state_schema import WriterState
from .strategy_layer import generate_beat_sheet
from .logic_layer import logic_check_beat_sheet, refine_beat_sheet_with_constraints
from .draft_layer import generate_draft_with_style
from .polish_layer import style_fingerprint_check
from .style_injector import StyleInjector
from .branch_simulation import get_three_plot_directions, generate_chapter_for_branch

try:
    from .graph_workflow import build_writer_graph, run_writer_flow
    from .graph_inspect import stream_writer_graph, run_with_inspection
except ImportError:
    build_writer_graph = None  # type: ignore
    run_writer_flow = None  # type: ignore
    stream_writer_graph = None  # type: ignore
    run_with_inspection = None  # type: ignore

__all__ = [
    "WriterState",
    "generate_beat_sheet",
    "logic_check_beat_sheet",
    "refine_beat_sheet_with_constraints",
    "generate_draft_with_style",
    "style_fingerprint_check",
    "StyleInjector",
    "get_three_plot_directions",
    "generate_chapter_for_branch",
    "build_writer_graph",
    "run_writer_flow",
    "stream_writer_graph",
    "run_with_inspection",
]
