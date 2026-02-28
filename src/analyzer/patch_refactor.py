# -*- coding: utf-8 -*-
"""
补丁隔离带 + 战略层重构（Refactor Cycle）：
- 低质量模型反馈先入 Pending_Patches，不直接改主模版。
- 每 N 章或补丁达阈值时，由高质量模型（架构师）压缩合并为模版 V2。
- 模版健康度：Token 数、空置率、补丁冲突检测。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from src.utils.llm_client import chat_high_quality

from .models import KnowledgeCard
from .state_schema import (
    ElementFieldDef,
    LogicRedLine,
    MetaProtocol,
    PendingPatch,
    UNCLASSIFIED_FIELD_NAME,
)

# 每处理完多少章可触发一次重构
REFACTOR_CHAPTER_INTERVAL = 50
# 补丁累积到多少条可触发重构
PATCH_THRESHOLD = 20
# 模版 Token 估算超过此值强制重构（约 1000 字符 ≈ 500 token，保守用 2000 字符）
HEALTH_TOKEN_CHAR_LIMIT = 2000
# 连续多少章空置率超阈值则考虑合并/删除字段
EMPTY_RATE_WINDOW = 5
EMPTY_RATE_THRESHOLD = 0.3
# 重构时最多带上的典型失败 Case 条数
SAMPLE_FAILURE_MAX = 5


def submit_patch(
    state: Any,
    chapter_id: str = "",
    chapter_index: int = 0,
    issue: str = "",
    suggestion: str = "",
) -> None:
    """将低质量模型反馈写入 Pending_Patches，不修改主模版。"""
    if not issue and not suggestion:
        return
    state.pending_patches = getattr(state, "pending_patches", []) or []
    state.pending_patches.append(
        PendingPatch(
            chapter_id=chapter_id,
            chapter_index=chapter_index,
            issue=(issue or "").strip(),
            suggestion=(suggestion or "").strip(),
        )
    )


def _protocol_token_estimate(protocol: Optional[MetaProtocol]) -> int:
    """粗略估算模版占用的字符数（用于健康检查）。"""
    if not protocol:
        return 0
    s = protocol.model_dump_json(exclude_none=True)
    return len(s)


def template_health_check(
    protocol: Optional[MetaProtocol],
    pending_count: int = 0,
    extraction_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    模版健康度检查。返回：
    - token_estimate: 模版估算字符数
    - over_token_limit: 是否超过 HEALTH_TOKEN_CHAR_LIMIT
    - over_patch_threshold: 是否达到 PATCH_THRESHOLD
    - empty_rate_suggestions: 若提供 extraction_stats，连续空置率过高的字段建议合并/删除
    - should_refactor: 是否建议触发重构
    """
    result: Dict[str, Any] = {
        "token_estimate": 0,
        "over_token_limit": False,
        "over_patch_threshold": pending_count >= PATCH_THRESHOLD,
        "should_refactor": pending_count >= PATCH_THRESHOLD,
        "empty_rate_suggestions": [],
    }
    if not protocol:
        return result
    result["token_estimate"] = _protocol_token_estimate(protocol)
    result["over_token_limit"] = result["token_estimate"] > HEALTH_TOKEN_CHAR_LIMIT
    if result["over_token_limit"]:
        result["should_refactor"] = True
    if extraction_stats and isinstance(extraction_stats.get("field_empty_rates"), dict):
        # 连续 N 章空置率 > 阈值的字段
        for field_name, rate in extraction_stats.get("field_empty_rates", {}).items():
            if isinstance(rate, (int, float)) and rate >= EMPTY_RATE_THRESHOLD:
                result["empty_rate_suggestions"].append(
                    f"字段「{field_name}」空置率 {rate:.0%}，考虑删除或合并"
                )
    return result


def refactor_meta_protocol_with_patches(
    protocol: MetaProtocol,
    patches: List[PendingPatch],
    sample_failures: Optional[List[str]] = None,
    book_id: str = "",
    title: str = "",
) -> MetaProtocol:
    """
    战略层「冷启动」重构：高质量模型（架构师）根据当前模版 V1 + 补丁 + 典型失败 Case，
    输出新模版 V2。要求：合并重复项，通过修改字段描述而非增加新字段，Token 控制在约 500 以内。
    """
    protocol_json = protocol.model_dump_json(exclude_none=True, indent=2)
    patches_text = "\n".join(
        f"- [ch{p.chapter_index}] {p.issue} → {p.suggestion}" for p in patches[:50]
    )
    failures_text = ""
    if sample_failures:
        failures_text = "\n".join(f"典型失败: {s}" for s in sample_failures[:SAMPLE_FAILURE_MAX])

    user = f"""你作为「元知识模版架构师」，请根据当前模版与补丁反馈，重写一份精简的元知识模版。

书名：{title}
book_id：{book_id}

## 当前模版 V1（JSON）
{protocol_json[:6000]}

## 补丁反馈（低质量模型未能识别的设定/建议）
{patches_text or "（无）"}

{failures_text}

要求：
1. 分析补丁共性，合并重复项。
2. 优先通过修改现有字段的 description 来覆盖新设定，而非增加新字段（字段抽象化：如用「技能系统」覆盖剑法/刀法/身法）。
3. 保留「{UNCLASSIFIED_FIELD_NAME}」作为通用溢出容器，用于新地图/新设定。
4. 输出模版总 Token 控制在约 500 以内（精简描述）。
5. 只输出一个合法 JSON 对象，包含 logic_red_lines、element_template、term_mapping、note，不要 markdown 包裹。
"""

    messages = [
        {"role": "system", "content": "你是元知识模版架构师，根据补丁与失败案例重写模版，只输出 JSON。"},
        {"role": "user", "content": user},
    ]
    raw = chat_high_quality(messages)
    raw = (raw or "").strip()
    for prefix in ("```json", "```"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):].lstrip()
            break
    if raw.endswith("```"):
        raw = raw[:-3].rstrip()
    data: Optional[Dict[str, Any]] = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(raw)):
                if raw[i] == "{":
                    depth += 1
                elif raw[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            data = json.loads(raw[start : i + 1])
                        except json.JSONDecodeError:
                            pass
                        break
    if not data or not isinstance(data, dict):
        return protocol

    red_lines = []
    for item in data.get("logic_red_lines") or []:
        if isinstance(item, dict):
            red_lines.append(LogicRedLine(
                category=item.get("category", ""),
                rule=item.get("rule", ""),
                source_chapter_ids=item.get("source_chapter_ids") or [],
            ))
    template = []
    for item in data.get("element_template") or []:
        if isinstance(item, dict) and item.get("name"):
            template.append(ElementFieldDef(
                name=item.get("name", ""),
                kind=item.get("kind", "str"),
                description=item.get("description", ""),
            ))
    if not template:
        template = list(protocol.element_template or [])
    term_mapping = data.get("term_mapping")
    if not isinstance(term_mapping, dict):
        term_mapping = dict(protocol.term_mapping or {})
    note = data.get("note", "") or f"由补丁重构 v{getattr(protocol, 'book_id', '')}"

    return MetaProtocol(
        book_id=book_id or protocol.book_id,
        logic_red_lines=red_lines,
        element_template=template,
        term_mapping=term_mapping,
        note=note,
    )


def deduplicate_and_resolve_conflicts(
    patches: List[PendingPatch],
) -> List[PendingPatch]:
    """
    高质量模型扫描补丁库，剔除互相矛盾的建议，合并重复项。
    返回去重、去冲突后的补丁列表。
    """
    if len(patches) <= 1:
        return list(patches)
    text = "\n".join(
        f"{i+1}. [ch{p.chapter_index}] issue: {p.issue}; suggestion: {p.suggestion}"
        for i, p in enumerate(patches[:80])
    )
    user = f"""以下是从低质量模型收集的补丁建议。请分析：
1. 合并重复或高度相似的条目。
2. 剔除互相矛盾的建议（例如一个说「增加字段A」、另一个说「不要提取A」）。
3. 只保留你认为应交给架构师合并进模版的条目。

补丁列表：
{text}

请只输出一个 JSON 数组，每项为 {{ "issue": "...", "suggestion": "..." }}，不要其他文字。若全部剔除则输出 []。
"""
    messages = [
        {"role": "system", "content": "你分析补丁列表，去重去冲突，只输出 JSON 数组。"},
        {"role": "user", "content": user},
    ]
    raw = chat_high_quality(messages)
    raw = (raw or "").strip()
    for prefix in ("```json", "```"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):].lstrip()
            break
    if raw.endswith("```"):
        raw = raw[:-3].rstrip()
    try:
        arr = json.loads(raw)
    except json.JSONDecodeError:
        return list(patches)
    if not isinstance(arr, list):
        return list(patches)
    seen = set()
    out: List[PendingPatch] = []
    for p in patches:
        key = (p.issue.strip(), p.suggestion.strip())
        if key in seen:
            continue
        for item in arr:
            if isinstance(item, dict) and item.get("issue") == p.issue and item.get("suggestion") == p.suggestion:
                seen.add(key)
                out.append(p)
                break
    return out if out else list(patches)


def try_refactor_if_needed(
    state: Any,
    current_chapter_index: int,
) -> Any:
    """
    检查是否达到重构阈值（章数间隔 / 补丁数 / 模版超 Token），若达到则执行战略层重构：
    去冲突 → 架构师合并为模版 V2 → 清空 pending_patches，主模版保持静态直到下次重构。
    """
    state.pending_patches = getattr(state, "pending_patches", []) or []
    state.last_refactor_at_chapter = getattr(state, "last_refactor_at_chapter", -1)
    state.template_version = getattr(state, "template_version", 1)
    health = template_health_check(
        getattr(state, "meta_protocol", None),
        pending_count=len(state.pending_patches),
    )
    chapters_since = current_chapter_index - state.last_refactor_at_chapter
    should = (
        health["over_token_limit"]
        or len(state.pending_patches) >= PATCH_THRESHOLD
        or (
            chapters_since >= REFACTOR_CHAPTER_INTERVAL
            and len(state.pending_patches) > 0
        )
    )
    if not should or not state.meta_protocol:
        return state
    patches = deduplicate_and_resolve_conflicts(state.pending_patches)
    sample_failures = [f"{p.issue} -> {p.suggestion}" for p in patches[:SAMPLE_FAILURE_MAX]]
    new_protocol = refactor_meta_protocol_with_patches(
        state.meta_protocol,
        patches,
        sample_failures=sample_failures,
        book_id=getattr(state, "book_id", ""),
        title=getattr(state, "title", ""),
    )
    state.meta_protocol = new_protocol
    state.pending_patches = []
    state.template_version = state.template_version + 1
    state.last_refactor_at_chapter = current_chapter_index
    try:
        from src.utils import get_logger
        get_logger().info(
            "模版重构完成: version=%s, last_refactor_at_chapter=%s",
            state.template_version, current_chapter_index,
        )
    except Exception:
        pass
    return state


def collect_patches_from_cards(
    cards: List[Any],
    chapter_id: str = "",
    chapter_index: int = 0,
) -> List[PendingPatch]:
    """
    从知识卡片的 attributes 中扫描「未分类设定」内容，生成待合并补丁建议。
    用于：溢出容器有内容时，说明低质量模型遇到了未识别设定，可写入 Pending_Patches。
    """
    result: List[PendingPatch] = []
    for c in cards or []:
        if not isinstance(c, KnowledgeCard):
            continue
        attrs = getattr(c, "attributes", None) or {}
        unclassified = attrs.get(UNCLASSIFIED_FIELD_NAME) or attrs.get("未分类设定")
        if not unclassified:
            continue
        if isinstance(unclassified, dict) and unclassified:
            for k, v in list(unclassified.items())[:5]:
                result.append(PendingPatch(
                    chapter_id=chapter_id,
                    chapter_index=chapter_index,
                    issue=f"未识别设定: {k}",
                    suggestion=f"通过扩展现有字段描述覆盖「{k}」或保留在未分类设定中",
                ))
        elif isinstance(unclassified, str) and unclassified.strip():
            result.append(PendingPatch(
                chapter_id=chapter_id,
                chapter_index=chapter_index,
                issue="未分类设定有文本内容",
                suggestion=unclassified.strip()[:200],
            ))
    return result
