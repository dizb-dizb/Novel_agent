# -*- coding: utf-8 -*-
"""
因果修复引擎（Mutation Propagator）：当数据库发生修改（如设定/角色变更）后，
自动将下游章节标记为 needs_review，并循环审查与重写细纲，消除逻辑断裂。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

try:
    from backend.engine.auto_novel_engine import AnalysisAgent
    from backend.engine.logic_master import LogicMasterAgent
    from backend.models.plot_graph import PlotNode, PlotGraphManager
    from backend.schemas.plot_schemas import NodeReviewRequest, NodeReviewResult
except ImportError:
    from engine.auto_novel_engine import AnalysisAgent
    from engine.logic_master import LogicMasterAgent
    from models.plot_graph import PlotNode, PlotGraphManager
    from schemas.plot_schemas import NodeReviewRequest, NodeReviewResult

logger = logging.getLogger(__name__)


class MutationPropagator:
    """
    逻辑审查大循环：查询所有 needs_review 的 PlotNode，按序调用 LogicMaster.review_node_logic，
    必要时覆写细纲并更新数据库状态，最后将 needs_review 置为 False。
    """

    def __init__(
        self,
        *,
        logic_master: LogicMasterAgent,
        analysis_agent: AnalysisAgent,
        get_db_snapshot: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.logic_master = logic_master
        self.analysis_agent = analysis_agent
        self._get_db_snapshot = get_db_snapshot

    async def _get_snapshot(self, book_id: str) -> Dict[str, Any]:
        """获取四大知识库快照；若未注入则使用默认实现。"""
        if self._get_db_snapshot:
            out = self._get_db_snapshot(book_id)
            return await out if hasattr(out, "__await__") else out
        try:
            from backend.engine.auto_novel_engine import assemble_rag_from_backend
            from backend.schemas.writing_schemas import WriteRequest
        except ImportError:
            from engine.auto_novel_engine import assemble_rag_from_backend
            from schemas.writing_schemas import WriteRequest
        # 用现有 RAG 逻辑组快照（简化：只取字符形式）
        from backend.database import AsyncSessionLocal
        from backend.models.character import Character
        from backend.models.plot_graph import PlotNode
        from sqlalchemy import select
        async with AsyncSessionLocal() as session:
            r = await session.execute(select(Character).where(Character.book_id == book_id))
            characters = [{"id": c.id, "name": c.name, "personality_profile": c.personality_profile} for c in (r.scalars().all() or [])]
            r2 = await session.execute(select(PlotNode).where(PlotNode.book_id == book_id).order_by(PlotNode.sequence_order))
            plot_events = [{"id": n.id, "summary": n.summary} for n in (r2.scalars().all() or [])]
        return {"characters": characters, "plot_events": plot_events, "settings": [], "relations": []}

    async def propagate_and_fix(self, book_id: str) -> int:
        """
        因果修复主循环：
        1) 查询所有 needs_review == True 的 PlotNode，按章节顺序排列。
        2) 对每个节点调用 LogicMaster.review_node_logic(node, current_db_snapshot)。
        3) 若返回需重写，则覆写该节点 summary，并将该节点 is_mutated 设为 True。
        4) 审查结束后将该节点 needs_review 设为 False。
        5) 若该节点被重写，调用 AnalysisAgent 更新一次数据库（用新细纲作为“正文”的替身，仅更新情节状态）。
        :return: 被重写的节点数量
        """
        try:
            from backend.database import AsyncSessionLocal
        except ImportError:
            from database import AsyncSessionLocal
        rewritten = 0
        async with AsyncSessionLocal() as session:
            manager = PlotGraphManager(session)
            nodes = await manager.get_nodes_needing_review(book_id)
            if not nodes:
                logger.info("无需要审查的节点: book_id=%s", book_id)
                return 0
            snapshot = await self._get_snapshot(book_id)
            for node in nodes:
                upstream_summary = ""
                if node.previous_node_id:
                    prev = await session.get(PlotNode, node.previous_node_id)
                    if prev and prev.is_mutated and prev.summary:
                        upstream_summary = f"上一章已修改，当前摘要: {prev.summary[:300]}..."
                request = NodeReviewRequest(
                    node_id=node.id,
                    current_summary=node.summary or "",
                    chapter_index=node.chapter_index,
                    involved_characters=node.involved_characters,
                    upstream_mutation_summary=upstream_summary,
                    db_snapshot=snapshot,
                )
                result: NodeReviewResult = await self.logic_master.review_node_logic(request)
                if result.should_rewrite and result.new_summary:
                    node.summary = result.new_summary
                    node.is_mutated = True
                    rewritten += 1
                    logger.info("节点 %s 细纲已重写: ch=%s", node.id, node.chapter_index)
                    await session.flush()
                    await self.analysis_agent.extract_and_update_db(
                        book_id,
                        node.chapter_index or 0,
                        result.new_summary,
                        session=session,
                    )
                node.needs_review = False
            await session.commit()
        return rewritten
