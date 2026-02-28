# -*- coding: utf-8 -*-
"""
整本仿写：全局状态与风格管理器（State & Style Models）。
作为各 Agent 之间传递信息的标准协议，使用 Pydantic 严格校验。
支持「全书逆向重构」：原著情节树 + 变异基调 → 适配细纲 → 渲染 → 建库。
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------- 原著章节节点（逆向重构输入） ----------


class OriginalChapterNode(BaseModel):
    """原著某一章的绝对客观情节，用于逆向重构时做同构映射。"""
    chapter_number: int = Field(..., ge=1, description="原著章节序号（1-based）")
    original_pov: str = Field(
        default="",
        description="原著本章视角人物/主角名",
    )
    original_events: List[str] = Field(
        default_factory=list,
        description="原著本章发生的事件列表（按顺序）",
    )
    original_goal: str = Field(
        default="",
        description="原著本章的爽点或目标，如「退婚打脸」「立三年之约」",
    )
    original_word_count: Optional[int] = Field(
        default=None,
        description="原著本章字数/字符数，用于仿写时满足原文篇幅；未提供时使用指纹平均篇幅",
    )


# ---------- 变异基调（用户输入的新设定） ----------


class MutationPremise(BaseModel):
    """变异基调：将原著世界观/角色映射到新设定下的规则。"""
    new_world_setting: str = Field(
        default="",
        description="新世界观描述，如：将修仙改为赛博朋克、古代改为星际机甲",
    )
    character_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="角色映射：原著角色名 -> 新书角色名，如 {'萧炎': 'V', '纳兰嫣然': '林娜'}",
    )
    core_rule_changes: List[str] = Field(
        default_factory=list,
        description="核心法则变动，如：斗气改为引擎马力、炼丹改为机甲核心调校、退婚改为解除商业联姻",
    )


# ---------- 风格指南（参考书文风） ----------


class StyleGuide(BaseModel):
    """风格指南：从参考书提取的文风特征，用于仿写时约束文笔与节奏。"""
    reference_book_name: str = Field(..., description="参考书名")
    vocabulary_features: List[str] = Field(
        default_factory=list,
        description="高频词汇与句式特征列表，如「杀气凛然」「一道寒光」",
    )
    pacing_rules: str = Field(
        default="",
        description="行文节奏规则，如「战斗描写需占三成」「开篇必有悬念钩子」",
    )
    dialogue_style: str = Field(
        default="",
        description="对话风格，如「简短有力、多口语」「古风半文半白」",
    )
    avg_chapter_length: Optional[float] = Field(
        default=None,
        description="参考书平均每章字数/字符数（来自文笔指纹），仿写时本章篇幅需尽量贴合",
    )


# ---------- 全书状态（生成进度） ----------


class BookState(BaseModel):
    """全书状态：记录当前写到了第几章、当前的主线目标。"""
    current_chapter: int = Field(..., ge=0, description="当前已写完的章序号（0-based 或 1-based 由调用方约定）")
    main_plot_goal: str = Field(default="", description="当前主线目标或本卷核心冲突")
    note: Optional[str] = Field(default=None, description="可选备注，如本卷名称")


# ---------- 单章设计图（逻辑层产出） ----------


class ChapterDesign(BaseModel):
    """单章设计图：逻辑审查/适配后产出的细纲，编写模块必须严格遵守。"""
    chapter_number: int = Field(..., ge=1, description="本章序号（1-based）")
    pov_character: str = Field(
        default="",
        description="视角人物（主角或关键角色名/ID；适配后为新设定下的名字）",
    )
    required_events: List[str] = Field(
        default_factory=list,
        description="本章必须发生的事件列表（简短描述）",
    )
    logic_constraints: List[str] = Field(
        default_factory=list,
        description="逻辑约束，如「主角目前重伤，不能使用高阶魔法」",
    )
    adapted_from_chapter: Optional[int] = Field(
        default=None,
        description="关联原著章节号（逆向重构时填写，表示本细纲由该原著章适配而来）",
    )


class BatchChapterDesign(BaseModel):
    """弧线级批量构思产出：一批 N 章的适配细纲，用于摊薄大模型推理耗时。"""
    designs: List[ChapterDesign] = Field(
        default_factory=list,
        description="本批次的章节设计图，按 chapter_number 升序",
    )


# ---------- 重构知识框架（整体搭建后落盘，供文本实现阶段使用） ----------


class ChapterMeta(BaseModel):
    """单章元信息：用于渲染阶段取篇幅等。"""
    chapter_number: int = Field(..., ge=1)
    original_goal: str = Field(default="")
    original_word_count: Optional[int] = Field(default=None)


class ReconstructedOutline(BaseModel):
    """整体搭建后的知识框架：全书 N 章的适配细纲，编写模块仅按此逐章实现文本。"""
    book_id: str = Field(..., description="书籍 ID")
    total_chapters: int = Field(..., ge=0, description="总章数")
    mutation_premise: Optional[MutationPremise] = Field(default=None)
    designs: List[ChapterDesign] = Field(
        default_factory=list,
        description="按章节序排列的适配细纲，1..N 章",
    )
    chapter_meta: List[ChapterMeta] = Field(
        default_factory=list,
        description="每章元信息（原著目标、篇幅等），与 designs 一一对应",
    )
