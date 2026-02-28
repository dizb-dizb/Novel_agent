# -*- coding: utf-8 -*-
"""
全局状态结构：承载「元协议」、因果链、冲突标记。
供双环分析流水线（抽样→协议生成→增量提取→高质量整合）与后续写作模块回溯使用。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .models import KnowledgeCard, PlotNode


# ---------- 元协议 (Meta-Protocol)：高质量模型产出 ----------


class LogicRedLine(BaseModel):
    """逻辑红线：不可违反的设定或规则。"""
    category: str = Field(default="", description="如：力量体系/角色底线/时间线")
    rule: str = Field(default="", description="具体规则描述")
    source_chapter_ids: List[str] = Field(default_factory=list)


class ElementFieldDef(BaseModel):
    """知识卡片要素模版中的单字段定义。"""
    name: str = Field(..., description="字段名，如 炼丹配方/境界/所属势力")
    kind: str = Field(default="str", description="str / list / dict / number")
    description: str = Field(default="", description="提取时的指导说明")


# 通用溢出容器字段名：低质量模型把「不属于已有字段的新奇设定」放入此处，避免补丁爆炸
UNCLASSIFIED_FIELD_NAME = "未分类设定"


def _default_element_template() -> List[ElementFieldDef]:
    """默认要素模版：含通用溢出容器，供新地图/新设定写入。"""
    return [
        ElementFieldDef(name="名称", kind="str", description="实体或概念名称"),
        ElementFieldDef(name="描述", kind="str", description="简要描述"),
        ElementFieldDef(name="首次出现章节", kind="str", description="chapter_id 或序号"),
        ElementFieldDef(
            name=UNCLASSIFIED_FIELD_NAME,
            kind="dict",
            description="提取本章出现的任何不属于已有字段的新奇设定、新地图、新概念，键值对形式。避免穷举字段。",
        ),
    ]


# 四大核心模板键名：指导低质量模型分策略提取
CORE_TEMPLATE_CHARACTER = "character"      # 角色
CORE_TEMPLATE_SETTING = "setting"          # 设定/世界观
CORE_TEMPLATE_ITEM_SCENE = "item_scene"    # 道具/场景
CORE_TEMPLATE_PLOT_EVENT = "plot_event"    # 情节/事件


class MetaProtocol(BaseModel):
    """元协议：针对本书的「分析指南」，由高质量模型从采样章节生成。"""
    logic_red_lines: List[LogicRedLine] = Field(default_factory=list)
    element_template: List[ElementFieldDef] = Field(
        default_factory=_default_element_template,
    )
    term_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="非规范名 -> 规范名，防止后续模型理解偏差",
    )
    book_id: str = Field(default="")
    note: str = Field(default="", description="协议备注")
    # 四大核心提取策略：AI 智能采样+优化产出，指导低质量模型按策略逐章提取
    core_templates: Dict[str, str] = Field(
        default_factory=dict,
        description="角色/设定/道具场景/情节事件 四类提取策略文案，key: character|setting|item_scene|plot_event",
    )


# ---------- 补丁隔离带 (Patch-Buffer)：不直接改主模版 ----------


class PendingPatch(BaseModel):
    """低质量模型反馈的未识别设定/字段建议，先入 Pending_Patches，由战略层定期合并。"""
    chapter_id: str = Field(default="", description="章节 id")
    chapter_index: int = Field(default=0, description="0-based 章节序号")
    issue: str = Field(default="", description="如：未能识别「隐藏境界」字段")
    suggestion: str = Field(default="", description="如：增加隐藏境界提取 → 架构师将合并为「境界」描述扩展")


# ---------- 冲突标记：高质量模型在整合时发现 ----------


class ConflictMark(BaseModel):
    """逻辑冲突标记，供后续改编做「逻辑纠偏」或死因解释。"""
    conflict_type: str = Field(default="", description="如 entity_resurrection / timeline / 设定矛盾")
    description: str = Field(default="")
    card_or_node_ids: List[str] = Field(default_factory=list)
    chapter_ids: List[str] = Field(default_factory=list)
    suggestion: str = Field(default="", description="纠偏建议")


# ---------- 小说数据库（整合后的结构化输出） ----------


# 实体类型与知识卡片类型的对应：设定、道具、场景、角色、事件
ENTITY_TYPE_设定 = "设定"
ENTITY_TYPE_道具 = "道具"
ENTITY_TYPE_场景 = "场景"
ENTITY_TYPE_角色 = "角色"
ENTITY_TYPE_事件 = "事件"


class NovelDatabase(BaseModel):
    """
    小说数据库：由高质量模型持续整合后的结构化知识库。
    包含设定/道具/场景/角色等核心实体、情节树、知识卡片与冲突标记。
    """
    book_id: str = Field(default="")
    title: str = Field(default="")

    # 按实体类型分组的核心元素（从 cards 归类）
    entities_by_type: Dict[str, List[KnowledgeCard]] = Field(
        default_factory=dict,
        description="设定、道具、场景、人物/角色、事件 等分类下的知识卡片列表",
    )

    # 情节树：节点 id -> PlotNode，形成因果链
    plot_tree: Dict[str, PlotNode] = Field(default_factory=dict)

    # 全部知识卡片（含未单独归类的）
    cards: List[KnowledgeCard] = Field(default_factory=list)

    # 冲突标记（逻辑矛盾等）
    conflict_marks: List[ConflictMark] = Field(default_factory=list)

    # 元协议（知识模板）引用
    meta_protocol: Optional[MetaProtocol] = None


def build_novel_database(state: AnalysisState) -> NovelDatabase:
    """
    从分析状态构建小说数据库：将 cards 按类型归类为设定/道具/场景/角色/事件等实体，并保留情节树。
    """
    type_map = {
        "设定": ENTITY_TYPE_设定,
        "人物": ENTITY_TYPE_角色,
        "角色": ENTITY_TYPE_角色,
        "地点": ENTITY_TYPE_场景,
        "场景": ENTITY_TYPE_场景,
        "物品": ENTITY_TYPE_道具,
        "道具": ENTITY_TYPE_道具,
        "事件": ENTITY_TYPE_事件,
    }
    entities_by_type: Dict[str, List[KnowledgeCard]] = {
        ENTITY_TYPE_设定: [],
        ENTITY_TYPE_道具: [],
        ENTITY_TYPE_场景: [],
        ENTITY_TYPE_角色: [],
        ENTITY_TYPE_事件: [],
    }
    for c in state.cards:
        t = (c.type or "").strip()
        key = type_map.get(t) or t or "设定"
        if key not in entities_by_type:
            entities_by_type[key] = []
        entities_by_type[key].append(c)
    return NovelDatabase(
        book_id=state.book_id,
        title=state.title,
        entities_by_type=entities_by_type,
        plot_tree=state.plot_tree,
        cards=state.cards,
        conflict_marks=state.conflict_marks,
        meta_protocol=state.meta_protocol,
    )


# ---------- 全局分析状态 ----------


class AnalysisState(BaseModel):
    """能容纳元协议、因果链和冲突标记的全局状态。"""
    book_id: str = Field(default="")
    title: str = Field(default="")

    # 元协议（第一阶段高质量模型产出）
    meta_protocol: Optional[MetaProtocol] = None

    # 采样结果：关键章节 id 或 index 列表，用于协议生成与后续对照
    sampled_chapter_ids: List[str] = Field(default_factory=list)
    sampled_chapter_indices: List[int] = Field(default_factory=list)

    # 因果链：剧情树节点 id -> PlotNode
    plot_tree: Dict[str, PlotNode] = Field(default_factory=dict)

    # 知识卡片列表（含人物/地点/物品/设定/事件）
    cards: List[KnowledgeCard] = Field(default_factory=list)

    # 冲突标记列表
    conflict_marks: List[ConflictMark] = Field(default_factory=list)

    # 当前处理进度（滑动窗口用）
    last_processed_chapter_index: int = Field(default=-1)

    # 补丁隔离带：低质量模型反馈先入此处，不直接改主模版
    pending_patches: List[PendingPatch] = Field(default_factory=list, description="Pending_Patches 表")
    template_version: int = Field(default=1, description="当前元协议版本号，重构后递增")
    last_refactor_at_chapter: int = Field(default=-1, description="上次触发重构时的章节索引")

    # 原始章节摘要缓存（可选，用于回溯时查来源）
    chapter_summaries: Dict[str, str] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


def create_initial_state(book_id: str, title: str = "") -> AnalysisState:
    """创建初始分析状态。"""
    return AnalysisState(book_id=book_id, title=title)


def save_state(state: AnalysisState, path: Path) -> None:
    """将分析状态持久化到 JSON 文件。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(state.model_dump_json(exclude_none=False, indent=2), encoding="utf-8")


def load_state(path: Path) -> Optional[AnalysisState]:
    """从 JSON 文件加载分析状态。"""
    p = Path(path)
    if not p.is_file():
        return None
    return AnalysisState.model_validate_json(p.read_text(encoding="utf-8"))
