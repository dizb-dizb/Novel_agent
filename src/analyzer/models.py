# -*- coding: utf-8 -*-
"""剧情节点与知识卡片的 Pydantic 模型。"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PlotNode(BaseModel):
    """剧情节点；每个节点应能关联到父节点形成因果链。"""
    id: str = Field(..., description="唯一 id")
    type: str = Field(default="scene", description="类型：scene / event / decision / 状态变更 / 关系位移 / 新设锚点")
    summary: str = Field(default="", description="摘要")
    chapter_id: Optional[str] = None
    chapter_index: Optional[int] = None
    parent_id: Optional[str] = None
    cause_effect_notes: str = Field(default="", description="与父节点的因果说明")


class KnowledgeCard(BaseModel):
    """知识卡片：Fact + Rule + Cause + Style 的封装。"""
    type: str = Field(..., description="人物 / 地点 / 物品 / 设定 / 事件")
    name: str = Field(default="", description="名称")
    description: str = Field(default="", description="描述")
    first_chapter_id: Optional[str] = None
    plot_node_ids: List[str] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict, description="元协议要素模版下的扩展字段")
