# -*- coding: utf-8 -*-
"""
高质量模型整合入口：每完成若干窗口的增量提取后，调用 refiner 做冲突检测与因果树合并。
可配置「每 N 个窗口触发一次」或「仅在发现异常时触发」。
"""
from typing import List, Optional

from .models import KnowledgeCard, PlotNode
from .refiner import detect_conflicts_and_merge
from .state_schema import AnalysisState, ConflictMark


def refine_after_window(
    state: AnalysisState,
    new_cards: List[KnowledgeCard],
    new_nodes: List[PlotNode],
    every_n_windows: int = 1,
    current_window_index: int = 0,
) -> tuple[AnalysisState, List[ConflictMark]]:
    """
    在每 N 个窗口后（或每窗口）执行一次高质量整合与冲突检测。
    :param every_n_windows: 每 N 个窗口调用一次 refiner；1 表示每窗口都调用
    """
    if (current_window_index + 1) % every_n_windows != 0 and (new_cards or new_nodes):
        # 仅合并，不调用高质量模型
        state.cards.extend(new_cards)
        for n in new_nodes:
            state.plot_tree[n.id] = n
        return state, []
    return detect_conflicts_and_merge(state, new_cards, new_nodes)
