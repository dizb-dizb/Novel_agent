# -*- coding: utf-8 -*-
"""
WriterAgent 状态机：供分层生成与 LangGraph 循环使用。
包含 beat_sheet、logic_check_report、draft、critique_feedback、style_samples、branches 等。
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WriterState(BaseModel):
    """写作 Agent 循环状态，用于分层生成与人机在环。"""
    book_id: str = Field(default="")
    chapter_index: int = Field(default=0, description="当前续写章节序号 0-based")
    user_intent: str = Field(default="", description="用户改写/续写意图")

    # 战略层输出：剧情节拍表（大纲与分镜）
    beat_sheet: str = Field(default="", description="本章节拍表，如：第1节…第2节…")
    chapter_type: str = Field(default="", description="章节类型：战斗章/感情章/日常章 等，用于动态 Prompt")

    # 逻辑层输出：因果合规后的细化大纲与约束边界
    logic_check_report: str = Field(default="", description="Logic_Check 报告：是否符合前文因果")
    constraint_boundaries: str = Field(default="", description="约束边界，如：主角需表现虚弱、对话偏冷淡")

    # 分支模拟：3 个可选走向
    plot_directions: List[Dict[str, Any]] = Field(default_factory=list, description="3 个可选剧情走向 [{summary, score}, ...]")
    selected_branch_index: int = Field(default=0, description="用户选中的走向索引 0/1/2")

    # 草稿层输出
    draft: str = Field(default="", description="风格化草稿正文 2000-3000 字")
    style_samples: List[Dict[str, Any]] = Field(default_factory=list, description="本次注入的风格样本摘要")

    # 润色层与批判节点
    polish_feedback: str = Field(default="", description="风格指纹对齐反馈")
    critique_feedback: str = Field(default="", description="Critique 节点反馈：崩人设/降智等，用于回退 Writer 或 Planner")

    # 三维上下文（由 Librarian 注入）
    history_causal_pack: str = Field(default="")
    new_causal_anchors: str = Field(default="")
    rule_constraints_pack: str = Field(default="")

    # 用户心理专家注入：偏好补丁与情感续写导航
    preference_patch: str = Field(default="", description="User Expert 产出，如：用户厌恶压抑不爆发，请 1500 字内安排反击")
    steering_hint: str = Field(default="", description="当前期待的走向，如：用户希望主角先隐忍")

    # 循环控制
    critique_passed: bool = Field(default=False, description="Critique 是否通过，通过则退出循环")
    max_rounds: int = Field(default=3, description="最多重试轮数")

    class Config:
        arbitrary_types_allowed = True
