# -*- coding: utf-8 -*-
"""WritingAgent 请求/响应数据结构：用户指令、关注角色、地点与 RAG 上下文沉淀。"""
from typing import List, Optional

from pydantic import BaseModel, Field


class WriteRequest(BaseModel):
    """续写/生成请求：用户指令 + 关注角色 + 发生地点。"""
    user_instruction: str = Field(..., description="用户指令，如「续写下一章」")
    focus_character_ids: List[str] = Field(default_factory=list, description="关注角色 ID 列表")
    location_id: Optional[str] = Field(default=None, description="发生地点 ID")


class WriteResponse(BaseModel):
    """生成响应：思维链 + 正文 + 本次使用的 RAG 上下文（用于数据沉淀）。"""
    thought_process: str = Field(default="", description="大模型思维链（CoT）")
    draft_content: str = Field(default="", description="正式网文内容")
    used_context: str = Field(default="", description="本次使用的 RAG 上下文，用于数据沉淀")
