# -*- coding: utf-8 -*-
"""
情节图谱：带因果链与变异标记的章节级 DAG/链表，支持「蝴蝶效应」追踪与修复。
升级自线性情节树，增加 previous_node_id/next_node_id 时序链表、is_mutated、needs_review。
"""
from __future__ import annotations

import uuid
from typing import Any, List, Optional

from sqlalchemy import Boolean, Column, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import relationship

try:
    from backend.database.database import Base
except ImportError:
    from database.database import Base


class PlotNode(Base):
    """
    情节节点（图谱版）：章节级节点，带时序链表与变异追踪。
    用于逆向全书 → 数据库重构 → 正向渲染的完整流水线。
    """

    __tablename__ = "plot_node"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    book_id = Column(String(64), nullable=True, index=True, comment="书籍 id")
    chapter_index = Column(Integer, nullable=True, index=True, comment="章节序号（1-based）")
    summary = Column(Text, nullable=True, comment="情节细纲/概要")
    involved_characters = Column(JSON, nullable=True, comment="参与角色，JSON 数组")
    outcomes = Column(JSON, nullable=True, comment="本章导致的状态改变，JSON")

    # 时序链表：章节间先后关系
    previous_node_id = Column(
        String(36),
        ForeignKey("plot_node.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="上一章节点 ID",
    )
    next_node_id = Column(
        String(36),
        ForeignKey("plot_node.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="下一章节点 ID",
    )

    # 变异与审查标记（蝴蝶效应）
    is_mutated = Column(Boolean, default=False, nullable=False, comment="该节点是否被人工或系统修改过")
    needs_review = Column(Boolean, default=False, nullable=False, comment="上游发生变异，需 LogicMaster 重新审查")

    sequence_order = Column(Integer, nullable=True, index=True, comment="全局顺位，用于排序")

    # ORM 关系（可选）
    previous_node = relationship("PlotNode", foreign_keys=[previous_node_id], remote_side=[id])
    next_node = relationship("PlotNode", foreign_keys=[next_node_id], remote_side=[id])

    def __repr__(self) -> str:
        return f"<PlotNode(id={self.id!r}, book_id={self.book_id!r}, ch={self.chapter_index}, needs_review={self.needs_review})>"


# ---------- 图谱工具：下游标记传播 ----------


class PlotGraphManager:
    """
    情节图谱管理：当某节点发生修改时，将其所有下游节点标记为 needs_review，
    以便 MutationPropagator 按序执行因果修复。
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def mark_downstream_needs_review(self, start_node_id: str) -> int:
        """
        从 start_node_id 出发，沿 next_node_id 遍历所有下游节点，将 needs_review 设为 True。
        :return: 被标记的节点数量
        """
        count = 0
        current_id: Optional[str] = start_node_id
        while current_id:
            node = await self.session.get(PlotNode, current_id)
            if node is None:
                break
            if not node.needs_review:
                node.needs_review = True
                count += 1
            current_id = node.next_node_id
        return count

    async def get_nodes_needing_review(self, book_id: str) -> List[PlotNode]:
        """按章节顺序返回该书所有 needs_review 为 True 的节点。"""
        from sqlalchemy import select
        result = await self.session.execute(
            select(PlotNode)
            .where(PlotNode.book_id == book_id, PlotNode.needs_review == True)
            .order_by(PlotNode.sequence_order.asc().nulls_last(), PlotNode.chapter_index.asc().nulls_last())
        )
        return list(result.scalars().all())

    async def get_head_node(self, book_id: str) -> Optional[PlotNode]:
        """返回该书链表的头节点（无 previous_node_id 的节点）。"""
        from sqlalchemy import select
        result = await self.session.execute(
            select(PlotNode).where(PlotNode.book_id == book_id, PlotNode.previous_node_id.is_(None)).limit(1)
        )
        return result.scalar_one_or_none()

    async def iter_nodes_in_order(self, book_id: str) -> List[PlotNode]:
        """按 previous/next 链表顺序返回该书所有节点。"""
        head = await self.get_head_node(book_id)
        if not head:
            from sqlalchemy import select
            result = await self.session.execute(
                select(PlotNode).where(PlotNode.book_id == book_id).order_by(PlotNode.sequence_order.asc().nulls_last(), PlotNode.chapter_index.asc().nulls_last())
            )
            return list(result.scalars().all())
        ordered: List[PlotNode] = []
        current: Optional[PlotNode] = head
        while current:
            ordered.append(current)
            if not current.next_node_id:
                break
            current = await self.session.get(PlotNode, current.next_node_id)
        return ordered
