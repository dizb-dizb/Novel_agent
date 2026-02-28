# -*- coding: utf-8 -*-
"""
低成本模型增量提取：在高质量模型给出的知识模板（元协议）指导下，
逐章提取知识卡片与剧情节点；整理整合由高质量模型在 pipeline 中完成。
"""
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from src.utils.llm_client import chat_high_quality, chat_low_cost

from .models import KnowledgeCard, PlotNode
from .state_schema import (
    CORE_TEMPLATE_CHARACTER,
    CORE_TEMPLATE_ITEM_SCENE,
    CORE_TEMPLATE_PLOT_EVENT,
    CORE_TEMPLATE_SETTING,
    MetaProtocol,
)


def _extract_json_from_response(raw: str) -> Optional[Dict[str, Any]]:
    """
    从 LLM 回复中尽量解析出 JSON。常见失败原因：带 markdown 代码块、前后有说明文字、换行/缩进问题。
    """
    if not raw or not raw.strip():
        return None
    s = raw.strip()
    for prefix in ("```json", "```"):
        if s.startswith(prefix):
            s = s[len(prefix):].lstrip()
            break
    if s.endswith("```"):
        s = s[:-3].rstrip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _normalize_extraction_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """兼容不同键名：模型可能返回英文键或中文键。"""
    if not data or not isinstance(data, dict):
        return data or {}
    out = {}
    cards_key = "knowledge_cards"
    nodes_key = "plot_nodes"
    if cards_key in data:
        out[cards_key] = data[cards_key]
    elif "知识卡片" in data:
        out[cards_key] = data["知识卡片"]
    else:
        out[cards_key] = []
    if nodes_key in data:
        out[nodes_key] = data[nodes_key]
    elif "剧情节点" in data:
        out[nodes_key] = data["剧情节点"]
    else:
        out[nodes_key] = []
    return out


def _protocol_prompt(protocol: Optional[MetaProtocol]) -> str:
    if not protocol:
        return "无元协议，请提取：人物、地点、物品、设定、事件及剧情节点。"
    parts = []
    # 四大核心提取策略（AI 采样+优化产出，指导低质量模型按策略提取）
    core = getattr(protocol, "core_templates", None) or {}
    if core:
        labels = {
            CORE_TEMPLATE_CHARACTER: "角色",
            CORE_TEMPLATE_SETTING: "设定/世界观",
            CORE_TEMPLATE_ITEM_SCENE: "道具/场景",
            CORE_TEMPLATE_PLOT_EVENT: "情节/事件",
        }
        parts.append("## 四大核心提取策略（请按下列策略逐类提取）")
        for key in (CORE_TEMPLATE_CHARACTER, CORE_TEMPLATE_SETTING, CORE_TEMPLATE_ITEM_SCENE, CORE_TEMPLATE_PLOT_EVENT):
            if key in core and core[key]:
                parts.append(f"### {labels.get(key, key)}\n{core[key]}")
    if protocol.logic_red_lines:
        parts.append("逻辑红线（不可违反）：")
        for r in protocol.logic_red_lines:
            parts.append(f"  - [{r.category}] {r.rule}")
    if protocol.element_template:
        parts.append("知识卡片字段：")
        for e in protocol.element_template:
            parts.append(f"  - {e.name} ({e.kind}): {e.description}")
    if protocol.term_mapping:
        parts.append("术语规范：")
        for k, v in protocol.term_mapping.items():
            parts.append(f"  - {k} -> {v}")
    return "\n".join(parts) if parts else "按通用要素提取。"


def _parse_extraction_result(
    data: dict,
    id_prefix: str,
    default_chapter_index: int = 0,
) -> Tuple[List[KnowledgeCard], List[PlotNode]]:
    """解析 LLM 返回的 JSON 为 KnowledgeCard / PlotNode 列表。"""
    cards: List[KnowledgeCard] = []
    nodes: List[PlotNode] = []
    for item in data.get("knowledge_cards") or []:
        if isinstance(item, dict):
            cards.append(KnowledgeCard(
                type=item.get("type", "设定"),
                name=item.get("name", ""),
                description=item.get("description", ""),
                first_chapter_id=item.get("first_chapter_id"),
                attributes=item.get("attributes") or {},
            ))
    for item in data.get("plot_nodes") or []:
        if isinstance(item, dict):
            nid = item.get("id") or (id_prefix + str(uuid.uuid4())[:8])
            if id_prefix and not nid.startswith(id_prefix):
                nid = id_prefix + nid.replace(" ", "-")
            nodes.append(PlotNode(
                id=nid,
                type=item.get("type", "event"),
                summary=item.get("summary", ""),
                chapter_id=item.get("chapter_id"),
                chapter_index=item.get("chapter_index", default_chapter_index),
                parent_id=item.get("parent_id"),
                cause_effect_notes=item.get("cause_effect_notes", ""),
            ))
    return cards, nodes


def extract_cards_from_chapter(
    chapter_title: str,
    content: str,
    chapter_id: str = "",
    chapter_index: int = 0,
    protocol: Optional[MetaProtocol] = None,
    existing_node_ids: Optional[List[str]] = None,
) -> Tuple[List[KnowledgeCard], List[PlotNode]]:
    """
    低成本模型：根据高质量模型给出的知识模板（元协议）从单章正文提取知识卡片与剧情节点。
    供 pipeline 逐章调用，提取结果由高质量模型在后续步骤中整理整合为整本书知识库。
    """
    text = (content or "")[:12000]
    protocol_text = _protocol_prompt(protocol)
    parent_hint = ""
    if existing_node_ids:
        parent_hint = f"已有剧情节点 id（可作父节点引用）：{existing_node_ids[:30]}"

    user = f"""请对以下单章内容做结构化提取。

{protocol_text}
{parent_hint}

## 本章正文

## 第 {chapter_index + 1} 章 [{chapter_id}] {chapter_title}

{text}

---

请严格输出一个 JSON 对象，且仅此 JSON（不要 markdown 代码块）。格式：

{{
  "knowledge_cards": [
    {{ "type": "人物|地点|物品|设定|事件", "name": "", "description": "", "first_chapter_id": "章节id", "attributes": {{}} }}
  ],
  "plot_nodes": [
    {{ "id": "建议 chapter_index-node_index", "type": "scene|event|decision|状态变更|关系位移|新设锚点", "summary": "摘要", "chapter_id": "", "chapter_index": {chapter_index}, "parent_id": "若可关联到前文节点则填已有节点id否则null", "cause_effect_notes": "与父节点的因果说明" }}
  ]
}}

要求：
1. 按协议中的知识卡片字段提取本章新出现或更新的实体与设定。
2. 每个 plot_node 尽量找到其「父节点」并填 parent_id 与 cause_effect_notes。
3. 若本章出现协议中未有专门字段的新设定、新地图、新概念，请放入该卡片的 attributes 的「未分类设定」键下（对象键值对），避免漏提。
"""

    messages = [
        {"role": "system", "content": "你是小说信息提取助手，只输出符合 schema 的 JSON，不输出其他内容。"},
        {"role": "user", "content": user},
    ]
    raw = chat_low_cost(messages)
    raw = (raw or "").strip()
    data = _extract_json_from_response(raw)
    # 低成本模型返回空或解析失败时，用高质量模型兜底一次，避免 novel_database 始终为空
    if data is None and (not raw or not raw.strip()):
        raw = chat_high_quality(messages)
        raw = (raw or "").strip()
        data = _extract_json_from_response(raw)
    if data is None:
        try:
            from src.utils import get_logger
            get_logger().warning(
                "extract_cards_from_chapter 解析失败 chapter_index=%s，原始回复前 500 字: %s",
                chapter_index, (raw or "")[:500],
            )
        except Exception:
            pass
        return [], []
    data = _normalize_extraction_data(data)
    prefix = f"ch{chapter_index}-"
    return _parse_extraction_result(data, prefix, chapter_index)


def extract_cards_from_window(
    window_chapters: List[dict],
    protocol: Optional[MetaProtocol],
    existing_node_ids: Optional[List[str]] = None,
    window_start_index: int = 0,
) -> Tuple[List[KnowledgeCard], List[PlotNode]]:
    """
    以 3 章为窗口，低成本模型在元协议指导下提取知识卡片与剧情节点。
    要求：识别状态变更、关系位移、新设锚点；剧情节点尽量标注 parent_id（因果链）。
    :param window_chapters: 3 章列表，每项含 chapter_id, chapter_title, content
    :param protocol: 元协议（可为 None）
    :param existing_node_ids: 已有剧情节点 id 列表，供模型回溯父节点
    :param window_start_index: 窗口起始章节序号（0-based）
    :return: (本窗口新卡片, 本窗口新节点)
    """
    text_parts = []
    for i, ch in enumerate(window_chapters):
        idx = window_start_index + i
        title = ch.get("chapter_title") or ""
        cid = ch.get("chapter_id") or ""
        content = (ch.get("content") or "")[:12000]
        text_parts.append(f"## 第 {idx + 1} 章 [{cid}] {title}\n\n{content}")
    combined = "\n\n---\n\n".join(text_parts)
    protocol_text = _protocol_prompt(protocol)
    parent_hint = ""
    if existing_node_ids:
        parent_hint = f"已有剧情节点 id（可作父节点引用）：{existing_node_ids[:30]}"

    user = f"""请对以下 3 章内容做结构化提取。

{protocol_text}
{parent_hint}

## 正文

{combined}

---

请严格输出一个 JSON 对象，且仅此 JSON（不要 markdown 代码块）。格式：

{{
  "knowledge_cards": [
    {{ "type": "人物|地点|物品|设定|事件", "name": "", "description": "", "first_chapter_id": "章节id", "attributes": {{}} }}
  ],
  "plot_nodes": [
    {{ "id": "唯一id建议用 chapter_index-node_index", "type": "scene|event|decision|状态变更|关系位移|新设锚点", "summary": "摘要", "chapter_id": "", "chapter_index": 0, "parent_id": "若可关联到前文节点则填已有节点id否则null", "cause_effect_notes": "与父节点的因果说明" }}
  ]
}}

要求：
1. 识别本单元内的状态变更、关系位移、新设锚点。
2. 每个 plot_node 尽量找到其「父节点」并填 parent_id 与 cause_effect_notes。
3. knowledge_cards 的 attributes 可填协议中定义的扩展字段。
"""

    messages = [
        {"role": "system", "content": "你是小说信息提取助手，只输出符合 schema 的 JSON，不输出其他内容。"},
        {"role": "user", "content": user},
    ]
    raw = chat_low_cost(messages)
    raw = (raw or "").strip()
    data = _extract_json_from_response(raw)
    if data is None:
        try:
            from src.utils import get_logger
            get_logger().warning(
                "extract_cards_from_window 解析失败 window_start=%s，原始回复前 500 字: %s",
                window_start_index, (raw or "")[:500],
            )
        except Exception:
            pass
        return [], []
    data = _normalize_extraction_data(data)
    prefix = f"w{window_start_index}-"
    return _parse_extraction_result(data, prefix, window_start_index)
