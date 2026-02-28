# -*- coding: utf-8 -*-
"""
动态上下文装载：续写时提供「三维上下文包」——
历史因果包、新因果锚点、规则约束包，供续写 Agent 与审核节点使用。
"""
from typing import Any, Dict, List, Optional

from src.analyzer.causal_tracker import RewriteImpactReport, assess_rewrite_impact
from src.analyzer.state_schema import AnalysisState


def build_rewrite_context(
    state: AnalysisState,
    rewritten_chapter_index: int,
    new_anchors_text: str,
    max_history_nodes: int = 50,
    max_history_cards: int = 80,
) -> Dict[str, Any]:
    """
    构建续写用的三维上下文包。
    :param state: 当前分析状态（含剧情树、卡片、元协议）
    :param rewritten_chapter_index: 被改写的章节序号（0-based）
    :param new_anchors_text: 用户改写后的核心变数摘要（或改写后章节的简要概括）
    :return: {
        "history_causal_pack": str,   # 未被改写影响的有效伏笔与因果
        "new_causal_anchors": str,    # 用户改写的新变数
        "rule_constraints_pack": str, # 世界观红线（元协议）
        "affected_summary": str,      # 改写影响报告摘要
    }
    """
    report = assess_rewrite_impact(state, rewritten_chapter_index)
    affected_ids = report.affected_node_ids

    # 历史因果包：chapter_index < rewritten 的节点与卡片，且不在受影响链上
    history_nodes = [
        n for n in state.plot_tree.values()
        if n.chapter_index is not None
        and n.chapter_index < rewritten_chapter_index
        and n.id not in affected_ids
    ]
    history_nodes = sorted(history_nodes, key=lambda x: (x.chapter_index or 0, x.id))[-max_history_nodes:]
    history_causal_pack = "\n".join(
        f"[ch{n.chapter_index}] {n.type}: {n.summary[:150]}"
        for n in history_nodes
    )

    history_cards = [
        c for c in state.cards
        if not c.plot_node_ids or not any(nid in affected_ids for nid in c.plot_node_ids)
    ][-max_history_cards:]
    history_causal_pack += "\n\n## 有效知识卡片\n"
    history_causal_pack += "\n".join(f"- [{c.type}] {c.name}: {c.description[:120]}" for c in history_cards)

    # 新因果锚点
    new_causal_anchors = new_anchors_text.strip() or "（无摘要，请以用户改写内容为准）"

    # 规则约束包：元协议中的逻辑红线
    rule_constraints_pack = ""
    if state.meta_protocol:
        rule_constraints_pack = "## 逻辑红线（不可违反）\n"
        for r in state.meta_protocol.logic_red_lines:
            rule_constraints_pack += f"- [{r.category}] {r.rule}\n"
        if state.meta_protocol.term_mapping:
            rule_constraints_pack += "\n## 术语规范\n"
            for k, v in list(state.meta_protocol.term_mapping.items())[:30]:
                rule_constraints_pack += f"- {k} -> {v}\n"
    if not rule_constraints_pack:
        rule_constraints_pack = "（无预研元协议，请勿违反前文已出现的设定）"

    return {
        "history_causal_pack": history_causal_pack,
        "new_causal_anchors": new_causal_anchors,
        "rule_constraints_pack": rule_constraints_pack,
        "affected_summary": report.summary,
        "report": report,
    }
