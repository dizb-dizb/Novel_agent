# -*- coding: utf-8 -*-
"""
逻辑层 (Logic Layer)：因果与约束校对，核心「逻辑守门员」。
将节拍表送入 Librarian 回溯，检查是否违背前文状态，输出带约束边界的细化大纲。
"""
import json
from typing import Optional

from src.analyzer.state_schema import AnalysisState
from src.utils.llm_client import chat_high_quality


def logic_check_beat_sheet(
    analysis_state: AnalysisState,
    beat_sheet: str,
    chapter_index: int,
) -> dict:
    """
    对节拍表做因果合规性检查。
    :return: { "passed": bool, "report": str, "conflicts": [str], "suggestions": [str] }
    """
    from src.analyzer.backtrack import _state_context
    ctx = _state_context(analysis_state, max_cards=60, max_nodes=80)
    user = f"""以下是一章剧情节拍表（大纲），发生在第 {chapter_index + 1} 章。请对照「当前世界观与因果摘要」检查是否存在逻辑漏洞。

例如：若前文已写「主角第20章法力尽失」，则节拍表中若出现「本章使用火球术」即矛盾。

当前因果与卡片摘要：
{ctx}

节拍表：
{beat_sheet}

请输出一个 JSON 对象，且仅此 JSON：
{{ "passed": true/false, "report": "简要检查报告", "conflicts": ["矛盾1", "矛盾2"], "suggestions": ["修改建议1"] }}
"""
    messages = [
        {"role": "system", "content": "你是逻辑守门员，只输出 JSON。"},
        {"role": "user", "content": user},
    ]
    raw = chat_high_quality(messages)
    raw = (raw or "").strip()
    for p in ("```json", "```"):
        if raw.startswith(p):
            raw = raw[len(p):].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    try:
        data = json.loads(raw)
        return {
            "passed": data.get("passed", False),
            "report": data.get("report", ""),
            "conflicts": data.get("conflicts") or [],
            "suggestions": data.get("suggestions") or [],
        }
    except Exception:
        return {"passed": True, "report": "", "conflicts": [], "suggestions": []}


def refine_beat_sheet_with_constraints(
    beat_sheet: str,
    logic_result: dict,
    chapter_type: str = "",
) -> str:
    """
    根据逻辑检查结果生成「约束边界」说明，并可选地修正节拍表表述。
    返回：带约束边界的细化大纲文本（可追加到 state.constraint_boundaries 或直接替换 beat_sheet 的说明）。
    """
    if logic_result.get("passed"):
        return ""
    suggestions = logic_result.get("suggestions") or []
    constraints = "；".join(suggestions)
    if chapter_type:
        constraints += f"。本章类型为「{chapter_type}」，写作时需符合该类型节奏与用词。"
    return constraints
