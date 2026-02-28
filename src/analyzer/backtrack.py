# -*- coding: utf-8 -*-
"""
逻辑回溯接口：供后续写作模块查询「在此时安排某剧情是否与既有知识卡片/因果链一致」。
例如：「如果我要在此时安排 A 杀掉 B，根据目前的知识卡片，是否合理？」
"""
import json
from typing import Any, Dict, List, Optional

from src.utils.llm_client import chat_high_quality

from .state_schema import AnalysisState


def _state_context(state: AnalysisState, max_cards: int = 80, max_nodes: int = 120) -> str:
    """构建供回溯查询的 state 摘要。"""
    parts = ["## 知识卡片"]
    for c in state.cards[-max_cards:]:
        parts.append(f"- [{c.type}] {c.name}: {c.description[:120]}")
    parts.append("\n## 剧情节点（因果）")
    for n in list(state.plot_tree.values())[-max_nodes:]:
        parts.append(f"- id={n.id} type={n.type} ch={n.chapter_index} parent_id={n.parent_id} summary={n.summary[:80]}")
    if state.conflict_marks:
        parts.append("\n## 已标记冲突")
        for m in state.conflict_marks:
            parts.append(f"- {m.conflict_type}: {m.description}")
    return "\n".join(parts)


def check_consistency(
    state: AnalysisState,
    action_description: str,
    at_chapter_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    检查「在给定位置安排某剧情」是否与当前知识卡片与因果链一致。
    :param state: 当前分析状态（含卡片、剧情树、冲突标记）
    :param action_description: 例如 "A 杀掉 B"、"主角在第 X 章获得某宝物"
    :param at_chapter_index: 可选，剧情发生的章节序号（0-based）
    :return: {
        "is_plausible": bool,
        "conflicts": [str],
        "suggestion": str,
        "related_cards": [str],
        "related_nodes": [str]
    }
    """
    ctx = _state_context(state)
    chapter_note = f"（发生章节序号：{at_chapter_index}）" if at_chapter_index is not None else ""

    user = f"""基于以下当前小说的知识卡片与剧情节点，判断「在此时安排以下剧情」是否合理、是否与已有设定/因果矛盾。

待判断剧情：{action_description} {chapter_note}

当前状态摘要：
{ctx}

请输出一个 JSON 对象，且仅此 JSON：
{{
  "is_plausible": true/false,
  "conflicts": ["矛盾1", "矛盾2"],
  "suggestion": "若不合理，给出纠偏或改写建议",
  "related_cards": ["与剧情相关的卡片名称或 id"],
  "related_nodes": ["与剧情相关的节点 id"]
}}
"""

    messages = [
        {"role": "system", "content": "你是小说逻辑顾问，根据已有设定与因果判断剧情是否合理，只输出 JSON。"},
        {"role": "user", "content": user},
    ]
    raw = chat_high_quality(messages)
    raw = raw.strip()
    for p in ("```json", "```"):
        if raw.startswith(p):
            raw = raw[len(p):].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "is_plausible": False,
            "conflicts": ["无法解析模型输出"],
            "suggestion": "",
            "related_cards": [],
            "related_nodes": [],
        }
    return {
        "is_plausible": data.get("is_plausible", False),
        "conflicts": data.get("conflicts") or [],
        "suggestion": data.get("suggestion") or "",
        "related_cards": data.get("related_cards") or [],
        "related_nodes": data.get("related_nodes") or [],
    }


def query_causal_parents(state: AnalysisState, node_id: str) -> List[str]:
    """查询某剧情节点的因果父节点链（用于写作时回溯历史节点）。"""
    node = state.plot_tree.get(node_id)
    if not node:
        return []
    path = []
    current_id = node.parent_id
    while current_id:
        path.append(current_id)
        parent = state.plot_tree.get(current_id)
        if not parent:
            break
        current_id = parent.parent_id
    return path
