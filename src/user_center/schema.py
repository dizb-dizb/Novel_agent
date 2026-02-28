# -*- coding: utf-8 -*-
"""
用户心理专家模块：数据模型与协议。
锚定用户爽点，转化为写作指令；支持偏好种子、情感续写导航、类似体验迁移。
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------- 爽点建模 (Trope Preference Mapping) ----------


class TropePreference(BaseModel):
    """单类爽点偏好：标签 + 强度 0-1。"""
    tag: str = Field(default="", description="如：绝境反杀/日常互动/幕后黑手/先抑后扬")
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    note: str = Field(default="", description="用户原话或行为依据")


class UserProfile(BaseModel):
    """用户画像：偏好维度，供 PsychologyExpert 产出「偏好补丁」。"""
    user_id: str = Field(default="", description="可选，多端同步用")
    book_id: str = Field(default="", description="当前锚定书籍，空表示全局偏好")

    # 节奏与情绪
    preferred_pacing: str = Field(default="medium", description="slow/medium/fast，节奏快慢")
    tragedy_tolerance: float = Field(default=0.5, ge=0.0, le=1.0, description="悲剧承受度，0 不接受 1 可接受虐")
    payoff_urgency: float = Field(default=0.6, ge=0.0, le=1.0, description="对「压抑后爆发」的迫切度，高则尽早安排反击")

    # 爽点指纹：多维度
    trope_preferences: List[TropePreference] = Field(default_factory=list)
    fetish_elements: List[str] = Field(default_factory=list, description="特殊癖好/要素，如：智斗、种田、群像")

    # 情感续写导航
    narrative_steering: str = Field(default="", description="当前期待的走向，如：希望主角现在就报仇 / 再隐忍一段")
    avoid_elements: List[str] = Field(default_factory=list, description="用户明确厌恶的桥段")

    # 元数据
    updated_at: str = Field(default="", description="ISO 或简单时间戳")
    source: str = Field(default="dialogue", description="dialogue / behavior / feedback")


class UserPreferenceProtocol(BaseModel):
    """
    用户偏好协议：持久化格式，对应 User_Preference_Protocol.json。
    第一阶段「偏好种子」产出，供后续生成前注入。
    """
    user_id: str = Field(default="")
    book_id: str = Field(default="")
    profile: UserProfile = Field(default_factory=UserProfile)
    raw_answers: Dict[str, Any] = Field(default_factory=dict, description="原始问答或行为摘要")
    version: str = Field(default="1.0")


# ---------- 实时上下文（供 Writer 注入） ----------


class UserContext(BaseModel):
    """
    运行时用户上下文：PsychologyExpert 产出，供生成模块使用。
    包含「偏好补丁」与「情感续写导航」。
    """
    preference_patch: str = Field(
        default="",
        description="注入 Writer 的偏好补丁，如：用户极度厌恶压抑不爆发，请在本章 1500 字内安排反击",
    )
    steering_hint: str = Field(default="", description="剧情走向提示，如：用户希望主角先隐忍")
    trope_weights: Dict[str, float] = Field(default_factory=dict, description="本章建议强化的爽点 tag -> 权重")
    hard_constraints: List[str] = Field(default_factory=list, description="硬性约束，如：不要出现 NTR")
    profile_snapshot: Optional[UserProfile] = None


# ---------- 意图解析结果 ----------


class ParsedIntent(BaseModel):
    """intent_analyzer 产出：用户当前续写/改写意图的结构化表示。"""
    intent_type: str = Field(default="continue", description="continue/rewrite/similar_book/ask/other")
    raw_text: str = Field(default="")
    summary: str = Field(default="", description="一句话摘要")
    slots: Dict[str, Any] = Field(default_factory=dict, description="槽位：如 target_chapter, focus_character")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


# ---------- 反馈记录（Satisfaction Tracker） ----------


class FeedbackRecord(BaseModel):
    """单条用户反馈。"""
    segment_id: str = Field(default="", description="段落或章节 id")
    book_id: str = Field(default="")
    chapter_index: int = Field(default=-1)
    rating: int = Field(default=0, description="1 赞 / -1 踩 / 0 未评")
    comment: str = Field(default="")
    created_at: str = Field(default="")


# ---------- 类似体验迁移（Recommendation） ----------


class TropeCard(BaseModel):
    """原书爽点结构卡片：供「换皮」推荐用。"""
    book_id: str = Field(default="")
    tropes: List[str] = Field(default_factory=list, description="如：孤独修行者、无敌幕后黑手")
    character_kernels: List[str] = Field(default_factory=list, description="人物性格内核")
    pacing: str = Field(default="medium")
    summary: str = Field(default="")


class WorldRuleCard(BaseModel):
    """原书世界观/规则摘要：换皮时保留逻辑、替换皮相。"""
    book_id: str = Field(default="")
    rules: List[str] = Field(default_factory=list)
    term_mapping: Dict[str, str] = Field(default_factory=dict)
    genre: str = Field(default="", description="修仙/科幻/都市 等")
