# -*- coding: utf-8 -*-
"""
因果追踪器：当第 N 章被改写时，在剧情树中递归查找受影响的后续节点与因果线。
用于「逻辑剪枝」：标记需冻结的节点与知识卡片，生成改写影响报告。
"""
from typing import List, Set, Tuple

from .models import KnowledgeCard, PlotNode
from .state_schema import AnalysisState


def get_ancestor_chain(state: AnalysisState, node_id: str) -> List[PlotNode]:
    """从某节点沿 parent_id 回溯到根，返回祖先节点列表（不含自身）。"""
    chain = []
    current_id = state.plot_tree.get(node_id).parent_id if state.plot_tree.get(node_id) else None
    while current_id:
        node = state.plot_tree.get(current_id)
        if not node:
            break
        chain.append(node)
        current_id = node.parent_id
    return chain


def get_nodes_in_chapter(state: AnalysisState, chapter_index: int) -> List[PlotNode]:
    """返回剧情树中属于第 chapter_index 章的所有节点（0-based）。"""
    return [
        n for n in state.plot_tree.values()
        if n.chapter_index is not None and n.chapter_index == chapter_index
    ]


def get_descendants_from_node_ids(
    state: AnalysisState,
    seed_node_ids: Set[str],
) -> Set[str]:
    """从一组种子节点出发，递归收集所有后代节点 id。"""
    descendants = set()
    current_layer = set(seed_node_ids)
    while current_layer:
        next_layer = set()
        for n in state.plot_tree.values():
            if n.parent_id and n.parent_id in current_layer:
                next_layer.add(n.id)
        descendants |= next_layer
        current_layer = next_layer
    return descendants


def get_affected_nodes_after_rewrite(
    state: AnalysisState,
    chapter_index: int,
) -> Tuple[List[PlotNode], Set[str]]:
    """
    当第 chapter_index 章被改写时，找出所有「以该章为因果上游」的后续节点。
    :return: (该章内的节点列表, 受影响的后续节点 id 集合)
    """
    nodes_in_chapter = get_nodes_in_chapter(state, chapter_index)
    seed_ids = {n.id for n in nodes_in_chapter}
    if not seed_ids:
        # 若无该章节点，则把所有 chapter_index > chapter_index 的节点视为可能受影响
        affected_ids = {
            n.id for n in state.plot_tree.values()
            if n.chapter_index is not None and n.chapter_index > chapter_index
        }
        return [], affected_ids
    affected_ids = get_descendants_from_node_ids(state, seed_ids)
    return nodes_in_chapter, affected_ids


def get_affected_cards_by_nodes(
    state: AnalysisState,
    affected_node_ids: Set[str],
) -> List[KnowledgeCard]:
    """根据受影响的剧情节点，找出关联的知识卡片（需复核或冻结）。"""
    out = []
    for c in state.cards:
        if not c.plot_node_ids:
            continue
        if any(nid in affected_node_ids for nid in c.plot_node_ids):
            out.append(c)
    return out


class RewriteImpactReport:
    """改写影响报告：供续写前逻辑剪枝与因果重构使用。"""
    def __init__(
        self,
        book_id: str,
        rewritten_chapter_index: int,
        nodes_in_rewritten_chapter: List[PlotNode],
        affected_node_ids: Set[str],
        affected_cards: List[KnowledgeCard],
        summary: str = "",
    ):
        self.book_id = book_id
        self.rewritten_chapter_index = rewritten_chapter_index
        self.nodes_in_rewritten_chapter = nodes_in_rewritten_chapter
        self.affected_node_ids = affected_node_ids
        self.affected_cards = affected_cards
        self.summary = summary


def assess_rewrite_impact(
    state: AnalysisState,
    chapter_index: int,
) -> RewriteImpactReport:
    """
    影响评估引擎：第 chapter_index 章变化时，遍历剧情树与卡片，生成改写影响报告。
    """
    nodes_in_ch, affected_ids = get_affected_nodes_after_rewrite(state, chapter_index)
    affected_cards = get_affected_cards_by_nodes(state, affected_ids)
    summary = (
        f"第 {chapter_index + 1} 章改写后，共 {len(affected_ids)} 个后续剧情节点、"
        f"{len(affected_cards)} 张关联知识卡片可能失效，建议冻结或重推。"
    )
    return RewriteImpactReport(
        book_id=state.book_id,
        rewritten_chapter_index=chapter_index,
        nodes_in_rewritten_chapter=nodes_in_ch,
        affected_node_ids=affected_ids,
        affected_cards=affected_cards,
        summary=summary,
    )
