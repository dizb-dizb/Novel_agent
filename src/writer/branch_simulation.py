# -*- coding: utf-8 -*-
"""
分支模拟：为用户提供 3 个可选剧情走向的摘要，选定后再生成对应全长章节。
"""
from typing import Any, Dict, List, Optional

from src.utils.llm_client import chat_high_quality

from .draft_layer import generate_draft_with_style
from .logic_layer import logic_check_beat_sheet, refine_beat_sheet_with_constraints
from .polish_layer import style_fingerprint_check
from .state_schema import WriterState
from .strategy_layer import generate_beat_sheet
from .style_injector import StyleInjector

try:
    from src.analyzer.state_schema import AnalysisState
except ImportError:
    AnalysisState = None  # type: ignore


def get_three_plot_directions(
    state: WriterState,
) -> List[Dict[str, Any]]:
    """
    根据用户意图与三维上下文，生成 3 个可选剧情走向的简短摘要。
    :return: [ {"summary": str, "score": None}, ... ]，score 由 Consultant 后续填写
    """
    ctx = f"""
## 历史因果（有效伏笔与节点）
{state.history_causal_pack[:2500]}

## 新因果锚点
{state.new_causal_anchors}

## 规则约束
{state.rule_constraints_pack[:1200]}
"""
    user = f"""你是一位网文策划。请根据以下信息，为「第 {state.chapter_index + 1} 章」设计 **3 个不同** 的剧情走向（每走向 2–4 句话摘要），供用户选择后再展开写全长。

用户意图：{state.user_intent}

{ctx}

要求：
1. 三个走向需在因果与规则约束内均成立，但情节发展明显不同（例如：A 潜入、B 正面冲突、C 暂时撤退）。
2. 输出格式严格为三块，每块以「走向1：」「走向2：」「走向3：」开头，后跟该走向的摘要。不要 JSON，不要其他解释。"""
    messages = [
        {"role": "system", "content": "你只输出三个走向的摘要，每块以「走向N：」开头。"},
        {"role": "user", "content": user},
    ]
    raw = chat_high_quality(messages)
    raw = (raw or "").strip()
    directions: List[Dict[str, Any]] = []
    for prefix in ["走向1：", "走向2：", "走向3："]:
        if prefix in raw:
            start = raw.index(prefix) + len(prefix)
            end = raw.find("走向", start) if start < len(raw) else len(raw)
            summary = raw[start:end].strip() if end > start else raw[start:].strip()
            directions.append({"summary": summary, "score": None})
    while len(directions) < 3:
        directions.append({"summary": "", "score": None})
    return directions[:3]


def generate_chapter_for_branch(
    state: WriterState,
    selected_branch_index: int,
    analysis_state: Optional["AnalysisState"] = None,
    style_injector: Optional[StyleInjector] = None,
    reference_chapter_length: Optional[int] = None,
) -> WriterState:
    """
    对用户选定的分支执行完整生成链：战略 → 逻辑 → 草稿 → 润色。
    若提供 reference_chapter_length，续写字数将与之相当（约 ±15%），便于与原文篇幅一致。
    会改写 state：beat_sheet, chapter_type, logic_check_report, constraint_boundaries, draft, style_samples, polish_feedback。
    """
    state.selected_branch_index = selected_branch_index
    # 用选定走向的摘要作为本轮的「意图」，驱动节拍表生成
    if state.plot_directions and 0 <= selected_branch_index < len(state.plot_directions):
        state.user_intent = state.plot_directions[selected_branch_index].get("summary", state.user_intent)

    # 战略层：节拍表 + 章节类型
    beat_sheet, chapter_type = generate_beat_sheet(state)
    state.beat_sheet = beat_sheet
    state.chapter_type = chapter_type

    # 逻辑层：因果校对（若有 analysis_state）
    if analysis_state is not None:
        logic_result = logic_check_beat_sheet(analysis_state, state.beat_sheet, state.chapter_index)
        state.logic_check_report = logic_result.get("report", "")
        state.constraint_boundaries = refine_beat_sheet_with_constraints(
            state.beat_sheet, logic_result, state.chapter_type
        )
    else:
        state.constraint_boundaries = f"本章类型为「{state.chapter_type}」，请按该类型节奏写作。"

    # 草稿层：Few-shot 风格生成（字数与参考章节相当）
    injector = style_injector or StyleInjector(book_id=state.book_id)
    draft, samples_used = generate_draft_with_style(
        state, style_injector=injector, reference_chapter_length=reference_chapter_length
    )
    state.draft = draft
    state.style_samples = samples_used

    # 润色层：风格指纹检查
    fingerprint = injector.style_store.get_fingerprint() if injector.style_store else None
    polish_passed, polish_feedback = style_fingerprint_check(state.draft, fingerprint)
    state.polish_feedback = polish_feedback

    return state
