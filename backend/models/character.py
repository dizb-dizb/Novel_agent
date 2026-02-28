# -*- coding: utf-8 -*-
"""角色表：姓名、别名、客观信息、性格与说话风格、向量库关联。"""
import uuid

from sqlalchemy import Column, JSON, String, Text

try:
    from backend.database.database import Base
except ImportError:
    from database.database import Base


class Character(Base):
    """角色表：用于网文角色档案与向量库语料关联。"""

    __tablename__ = "character"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    book_id = Column(String(64), nullable=True, index=True, comment="来源书籍 id，用于按书隔离")
    name = Column(String(255), nullable=False, index=True, comment="姓名")
    aliases = Column(JSON, nullable=True, comment="别名，JSON 数组如 [\"别名1\", \"别名2\"]")
    basic_info = Column(JSON, nullable=True, comment="客观信息，JSON 如 {年龄, 门派, 境界, ...}")
    personality_profile = Column(Text, nullable=True, comment="性格深度分析摘要")
    speaking_style = Column(Text, nullable=True, comment="说话习惯与口头禅摘要")
    embedding_id = Column(String(255), nullable=True, index=True, comment="向量库中该角色详细语料切片的 ID")

    def __repr__(self) -> str:
        return f"<Character(id={self.id!r}, name={self.name!r})>"
