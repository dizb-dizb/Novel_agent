# -*- coding: utf-8 -*-
"""角色相关 Pydantic 校验模型：Create / Read / Update。"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------- 公共字段 ----------
class CharacterBase(BaseModel):
    """角色公共字段（创建与更新共用）。"""
    name: str = Field(..., min_length=1, max_length=255, description="姓名")
    aliases: Optional[List[str]] = Field(default=None, description="别名列表")
    basic_info: Optional[Dict[str, Any]] = Field(default=None, description="客观信息，如年龄、门派、境界等")
    personality_profile: Optional[str] = Field(default=None, description="性格深度分析摘要")
    speaking_style: Optional[str] = Field(default=None, description="说话习惯与口头禅摘要")
    embedding_id: Optional[str] = Field(default=None, max_length=255, description="向量库中该角色语料切片 ID")


# ---------- Create ----------
class CharacterCreate(CharacterBase):
    """创建角色请求体（id 由服务端生成）。"""
    pass


# ---------- Read ----------
class CharacterRead(CharacterBase):
    """角色查询响应（含主键）。"""
    id: str = Field(..., description="主键 UUID")

    model_config = {"from_attributes": True}


# ---------- Update ----------
class CharacterUpdate(BaseModel):
    """更新角色请求体（全部可选，只提交要改的字段）。"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    aliases: Optional[List[str]] = None
    basic_info: Optional[Dict[str, Any]] = None
    personality_profile: Optional[str] = None
    speaking_style: Optional[str] = None
    embedding_id: Optional[str] = Field(default=None, max_length=255)
