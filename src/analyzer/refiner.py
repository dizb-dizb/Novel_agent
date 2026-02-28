# -*- coding: utf-8 -*-
"""
高质量模型：实体对齐与冲突检测。
对比低成本模型提取的新卡片/节点与已有状态，发现逻辑矛盾并生成 ConflictMark；
将新内容合并进全局状态（因果树、卡片列表）。
"""
import json
from typing import List, Tuple

from src.utils.llm_client import chat_high_quality, chat_medium_quality

from .models import KnowledgeCard, PlotNode
from .state_schema import AnalysisState, ConflictMark


def _state_summary(state: AnalysisState, max_cards: int = 50, max_nodes: int = 80) -> str:
    """生成供 LLM 参考的已有状态摘要。"""
    parts = ["## 已有知识卡片（部分）"]
    for c in state.cards[-max_cards:]:
        parts.append(f"- [{c.type}] {c.name}: {c.description[:80]}...")
    parts.append("\n## 已有剧情节点（部分）")
    for n in state.plot_tree.values():
        if len(parts) > max_cards + max_nodes + 10:
            break
        parts.append(f"- id={n.id} type={n.type} summary={n.summary[:60]} parent_id={n.parent_id}")
    return "\n".join(parts)


def detect_conflicts_and_merge(
    state: AnalysisState,
    new_cards: List[KnowledgeCard],
    new_nodes: List[PlotNode],
) -> Tuple[AnalysisState, List[ConflictMark]]:
    """
    高质量模型：对比新卡片/节点与已有状态，检测冲突并合并。
    :return: (更新后的 state, 本轮新增的 ConflictMark 列表)
    """
    if not new_cards and not new_nodes:
        return state, []

    new_cards_text = "\n".join(
        f"- [{c.type}] {c.name}: {c.description[:200]}" for c in new_cards
    )
    new_nodes_text = "\n".join(
        f"- id={n.id} type={n.type} summary={n.summary[:150]} parent_id={n.parent_id}"
        for n in new_nodes
    )
    existing = _state_summary(state)

    user = f"""现有状态摘要：
{existing}

本轮新增知识卡片：
{new_cards_text}

本轮新增剧情节点：
{new_nodes_text}

请检查：是否存在与已有状态矛盾的逻辑（例如角色已在前面章节死亡却又出现、时间线冲突、设定矛盾等）。
若存在矛盾，请输出一个 JSON 对象，包含 key "conflicts"，值为数组，每项为：
{{ "conflict_type": "类型", "description": "描述", "card_or_node_ids": ["id"], "chapter_ids": [], "suggestion": "纠偏建议" }}
若无矛盾，请输出 {{ "conflicts": [] }}。
只输出此 JSON，不要其他内容。
"""

    messages = [
        {"role": "system", "content": "你是小说逻辑校验助手，只输出 JSON。"},
        {"role": "user", "content": user},
    ]
    # 整合阶段使用中质量模型，未配置 ANALYZER_MEDIUM_MODEL 时用高质量模型
    raw = chat_medium_quality(messages)
    raw = raw.strip()
    for p in ("```json", "```"):
        if raw.startswith(p):
            raw = raw[len(p):].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    try:
        data = json.loads(raw)
        conflicts_data = data.get("conflicts") or []
    except Exception:
        conflicts_data = []

    new_conflicts: List[ConflictMark] = []
    for c in conflicts_data:
        if isinstance(c, dict):
            new_conflicts.append(ConflictMark(
                conflict_type=c.get("conflict_type", ""),
                description=c.get("description", ""),
                card_or_node_ids=c.get("card_or_node_ids") or [],
                chapter_ids=c.get("chapter_ids") or [],
                suggestion=c.get("suggestion", ""),
            ))

    # 合并：新卡片与节点加入 state
    state.cards.extend(new_cards)
    for n in new_nodes:
        state.plot_tree[n.id] = n
    state.conflict_marks.extend(new_conflicts)
    return state, new_conflicts
