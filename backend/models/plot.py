# -*- coding: utf-8 -*-
"""
情节树：小说宏观与微观事件因果，树状结构（父子节点）。
Volume 卷 -> Chapter 章 -> Scene 场景 -> Action 微观动作。
"""
import enum
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, Enum, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

try:
    from backend.database.database import Base
except ImportError:
    from database.database import Base


class EventLevel(str, enum.Enum):
    """事件层级：卷 / 章 / 场景 / 微观动作。"""
    Volume = "Volume"   # 卷
    Chapter = "Chapter" # 章
    Scene = "Scene"    # 场景
    Action = "Action"   # 微观动作


class Event(Base):
    """
    情节树节点：记录小说的宏观与微观事件因果。
    通过 parent_id 自引用形成树，level 表示粒度。
    """

    __tablename__ = "plot_event"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    book_id = Column(String(64), nullable=True, index=True, comment="来源书籍 id，用于按书隔离")
    parent_id = Column(
        String(36),
        ForeignKey("plot_event.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="父节点事件 ID",
    )
    level = Column(
        Enum(EventLevel),
        nullable=False,
        index=True,
        comment="层级：Volume/Chapter/Scene/Action",
    )
    summary = Column(Text, nullable=True, comment="情节概要（低质量模型总结）")
    involved_characters = Column(
        JSON,
        nullable=True,
        comment="参与角色 ID 列表，JSON 数组",
    )
    outcomes = Column(
        JSON,
        nullable=True,
        comment="事件导致的状态改变，JSON 如 {道具获得, 境界提升, ...}",
    )
    sequence_order = Column(
        Integer,
        nullable=True,
        index=True,
        comment="剧情顺位排序号，用于同层兄弟排序",
    )

    # 自引用关系（可选，便于 ORM 遍历）
    parent = relationship("Event", remote_side=[id], backref="children")

    def __repr__(self) -> str:
        return f"<Event(id={self.id!r}, level={self.level!r}, summary={self.summary[:30] if self.summary else None!r}...)>"


# ---------- 情节脉络追溯 ----------

async def get_plot_lineage(
    session: Any,
    event_id: str,
) -> List[Event]:
    """
    根据给定事件 ID 向上追溯，返回从该事件到 Volume 的完整剧情脉络（含自身与根卷）。
    顺序：当前事件 -> 父 -> ... -> Volume 节点。

    :param session: SQLAlchemy 异步会话（AsyncSession）
    :param event_id: 起始事件 ID
    :return: 事件列表 [当前事件, 父事件, ..., Volume]，若未到 Volume 则到 root 为止
    """
    lineage: List[Event] = []
    current_id: Optional[str] = event_id

    while current_id:
        result = await session.get(Event, current_id)
        if result is None:
            break
        lineage.append(result)
        if result.level == EventLevel.Volume:
            break
        current_id = result.parent_id

    return lineage
