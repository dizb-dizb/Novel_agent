# -*- coding: utf-8 -*-
"""
改写与续写闭环：从「用户改写第 N 章」到「影响评估 → 三维上下文 → 逻辑骨架/续写 → 双重校对 → 增量同步」。
可与 LangGraph 对接：将各步封装为节点，组成 SimulationGraph。
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

from .causal_tracker import RewriteImpactReport, assess_rewrite_impact
from .double_check import double_check_gate
from .state_schema import AnalysisState, load_state, save_state


def step1_impact_assessment(
    state: AnalysisState,
    chapter_index: int,
) -> RewriteImpactReport:
    """第一步：改写影响评估，生成需冻结/重推的因果线与卡片摘要。"""
    return assess_rewrite_impact(state, chapter_index)


def step2_dynamic_context(
    state: AnalysisState,
    rewritten_chapter_index: int,
    new_anchors_text: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """第二步：动态上下文装载，得到三维上下文包。"""
    from src.librarian.context_loader import build_rewrite_context
    return build_rewrite_context(
        state,
        rewritten_chapter_index,
        new_anchors_text,
        **kwargs,
    )


def step3_double_check(
    state: AnalysisState,
    draft_text: str,
    chapter_index: int,
    style_samples: Optional[List[Any]] = None,
    style_threshold: float = 0.5,
) -> Dict[str, Any]:
    """第三步：逻辑与风格双重校对。"""
    return double_check_gate(
        state,
        draft_text,
        chapter_index,
        style_samples=style_samples,
        style_threshold=style_threshold,
    )


def step4_incremental_sync(
    state: AnalysisState,
    new_chapter_content: str,
    new_chapter_title: str,
    chapter_index: int,
    chapter_id: str = "",
) -> AnalysisState:
    """
    第四步：用户确认新章后，用低成本模型对新章节做知识卡片提取并更新 state，
    使「虚拟剧情」转正，保证后续因果回溯一致。
    """
    from .extractor import extract_cards_from_chapter
    from .refiner import detect_conflicts_and_merge

    existing_ids = list(state.plot_tree.keys())
    cards, nodes = extract_cards_from_chapter(
        new_chapter_title,
        new_chapter_content,
        chapter_id=chapter_id or str(chapter_index),
        chapter_index=chapter_index,
        protocol=state.meta_protocol,
        existing_node_ids=existing_ids,
    )
    state, _ = detect_conflicts_and_merge(state, cards, nodes)
    return state


def run_rewrite_flow(
    state: AnalysisState,
    chapter_index: int,
    new_anchors_text: str,
    draft_text_for_check: str = "",
    style_samples: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    串联执行：影响评估 → 上下文装载 →（若提供 draft）双重校对。
    返回各步结果，供上游续写 Agent 或 LangGraph 使用。
    """
    report = step1_impact_assessment(state, chapter_index)
    ctx_pack = step2_dynamic_context(state, chapter_index, new_anchors_text)
    check_result = None
    if draft_text_for_check.strip():
        check_result = step3_double_check(
            state,
            draft_text_for_check,
            chapter_index,
            style_samples=style_samples,
        )
    return {
        "impact_report": report,
        "context_pack": ctx_pack,
        "double_check": check_result,
    }


def load_state_for_rewrite(book_id: str, cards_dir: Optional[Path] = None) -> Optional[AnalysisState]:
    """从 data/cards/{book_id}/analysis_state.json 加载状态，供改写流使用。"""
    from pathlib import Path
    base = cards_dir or Path(__file__).resolve().parents[2] / "data" / "cards"
    path = base / book_id / "analysis_state.json"
    return load_state(path)
