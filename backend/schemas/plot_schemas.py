# -*- coding: utf-8 -*-
"""
情节图谱相关 Pydantic 模式：逻辑审查请求、节点 DTO 等。
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------- 节点 DTO（与 ORM 对应） ----------


class PlotNodeCreate(BaseModel):
    """创建情节节点请求。"""
    book_id: Optional[str] = None
    chapter_index: Optional[int] = None
    summary: Optional[str] = None
    involved_characters: Optional[List[str]] = None
    outcomes: Optional[Dict[str, Any]] = None
    previous_node_id: Optional[str] = None
    next_node_id: Optional[str] = None
    is_mutated: bool = False
    needs_review: bool = False
    sequence_order: Optional[int] = None


class PlotNodeRead(BaseModel):
    """情节节点只读视图。"""
    id: str
    book_id: Optional[str] = None
    chapter_index: Optional[int] = None
    summary: Optional[str] = None
    involved_characters: Optional[List[str]] = None
    outcomes: Optional[Dict[str, Any]] = None
    previous_node_id: Optional[str] = None
    next_node_id: Optional[str] = None
    is_mutated: bool = False
    needs_review: bool = False
    sequence_order: Optional[int] = None

    class Config:
        from_attributes = True


# ---------- 逻辑审查请求（MutationPropagator → LogicMaster） ----------


class NodeReviewRequest(BaseModel):
    """
    节点逻辑审查请求：当前节点内容 + 上游变异摘要 + 当前数据库快照。
    供 LogicMaster.review_node_logic 使用。
    """
    node_id: str = Field(..., description="当前待审查节点 ID")
    current_summary: str = Field(default="", description="当前章节细纲/概要")
    chapter_index: Optional[int] = Field(default=None, description="章节序号")
    involved_characters: Optional[List[str]] = Field(default=None, description="当前节点涉及角色")
    upstream_mutation_summary: str = Field(
        default="",
        description="上游剧情变异摘要（如：某角色已删除、世界观已改为赛博朋克）",
    )
    db_snapshot: Dict[str, Any] = Field(
        default_factory=dict,
        description="当前四大知识库快照（角色、设定、关系、情节）",
    )


class NodeReviewResult(BaseModel):
    """LogicMaster 审查结果：是否需要重写 + 新细纲（若重写）。"""
    should_rewrite: bool = Field(default=False, description="是否需重写该章细纲")
    new_summary: Optional[str] = Field(default=None, description="重写后的细纲（若 should_rewrite 为 True）")
    logic_notes: str = Field(default="", description="审查说明（为何一致或为何需改）")
