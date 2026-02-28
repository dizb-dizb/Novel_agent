# -*- coding: utf-8 -*-
"""
将分析结果（novel_database：角色卡片 + 情节树）写入 SQLite。
供 analyze 完成后调用或通过 API 触发。
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

try:
    from backend.models.character import Character
    from backend.models.plot import Event, EventLevel
except ImportError:
    from models.character import Character
    from models.plot import Event, EventLevel

# 角色类卡片类型（写入 Character 表）
CHARACTER_CARD_TYPES = ("人物", "角色")


def _level_from_plot_node_type(node_type: str) -> EventLevel:
    """将 PlotNode.type 映射为 EventLevel。"""
    t = (node_type or "").strip().lower()
    if t in ("volume", "卷"):
        return EventLevel.Volume
    if t in ("chapter", "章"):
        return EventLevel.Chapter
    if t in ("scene", "场景"):
        return EventLevel.Scene
    if t in ("event", "decision", "状态变更", "关系位移", "新设锚点", "微观动作"):
        return EventLevel.Action
    return EventLevel.Scene


async def write_characters_from_novel_db(
    session: AsyncSession,
    novel_db: Dict[str, Any],
    book_id: str,
    *,
    upsert_by_name: bool = True,
) -> int:
    """
    从 novel_database 的 cards / entities_by_type 中写入角色到 Character 表。
    只处理 type 为 人物/角色 的卡片。
    :return: 写入条数
    """
    book_id = (book_id or "").strip() or (novel_db.get("book_id") or "")
    cards: List[Dict[str, Any]] = []
    entities = novel_db.get("entities_by_type") or {}
    for key in ("角色", "人物"):
        for c in entities.get(key) or []:
            if isinstance(c, dict):
                cards.append(c)
    raw_cards = novel_db.get("cards") or []
    for c in raw_cards:
        if not isinstance(c, dict):
            continue
        t = (c.get("type") or "").strip()
        if t in CHARACTER_CARD_TYPES and c not in cards:
            cards.append(c)

    seen_names: set = set()
    count = 0
    for c in cards:
        name = (c.get("name") or "").strip()
        if not name:
            continue
        if upsert_by_name and name in seen_names:
            continue
        seen_names.add(name)

        attrs = c.get("attributes") or {}
        if isinstance(attrs, str):
            attrs = {}
        aliases = attrs.get("别名") or attrs.get("aliases")
        if isinstance(aliases, str):
            aliases = [aliases] if aliases else None
        basic_info = {k: v for k, v in attrs.items() if k not in ("别名", "aliases", "未分类设定")}
        if not basic_info:
            basic_info = None
        personality_profile = (c.get("description") or "").strip() or None
        speaking_style = (attrs.get("说话习惯") or attrs.get("speaking_style") or "").strip() or None
        embedding_id = (attrs.get("embedding_id") or c.get("embedding_id") or "").strip() or None

        if upsert_by_name:
            result = await session.execute(
                select(Character).where(
                    Character.book_id == book_id,
                    Character.name == name,
                ).limit(1)
            )
            existing = result.scalar_one_or_none()
            if existing:
                existing.aliases = aliases
                existing.basic_info = basic_info
                existing.personality_profile = personality_profile
                existing.speaking_style = speaking_style
                existing.embedding_id = embedding_id
                count += 1
                continue
        char = Character(
            id=str(uuid.uuid4()),
            book_id=book_id or None,
            name=name,
            aliases=aliases,
            basic_info=basic_info,
            personality_profile=personality_profile,
            speaking_style=speaking_style,
            embedding_id=embedding_id,
        )
        session.add(char)
        count += 1
    return count


async def write_events_from_novel_db(
    session: AsyncSession,
    novel_db: Dict[str, Any],
    book_id: str,
) -> int:
    """
    从 novel_database 的 plot_tree 写入情节树到 Event 表。
    :return: 写入条数
    """
    book_id = (book_id or "").strip() or (novel_db.get("book_id") or "")
    plot_tree = novel_db.get("plot_tree") or {}
    if not isinstance(plot_tree, dict):
        return 0

    count = 0
    for node_id, node in plot_tree.items():
        if not isinstance(node, dict):
            continue
        summary = (node.get("summary") or "").strip() or None
        parent_id = (node.get("parent_id") or "").strip() or None
        node_type = (node.get("type") or "scene").strip()
        level = _level_from_plot_node_type(node_type)
        chapter_index = node.get("chapter_index")
        if chapter_index is None:
            chapter_index = 0
        try:
            sequence_order = int(chapter_index)
        except (TypeError, ValueError):
            sequence_order = 0
        involved_characters = node.get("involved_characters")
        if not isinstance(involved_characters, list):
            involved_characters = None
        outcomes = node.get("outcomes")
        if not isinstance(outcomes, dict):
            outcomes = None

        eid = str(node_id)
        existing = await session.get(Event, eid)
        if existing:
            existing.book_id = book_id or None
            existing.parent_id = parent_id
            existing.level = level
            existing.summary = summary
            existing.involved_characters = involved_characters
            existing.outcomes = outcomes
            existing.sequence_order = sequence_order
        else:
            session.add(Event(
                id=eid,
                book_id=book_id or None,
                parent_id=parent_id,
                level=level,
                summary=summary,
                involved_characters=involved_characters,
                outcomes=outcomes,
                sequence_order=sequence_order,
            ))
        count += 1
    return count


async def write_novel_database_to_db(
    session: AsyncSession,
    novel_db: Dict[str, Any],
    book_id: Optional[str] = None,
    *,
    write_characters: bool = True,
    write_events: bool = True,
) -> Dict[str, int]:
    """
    将 novel_database（分析产出）写入 SQLite：角色表 + 情节树表。
    :param session: 异步会话，调用方负责 commit/rollback。
    :param novel_db: novel_database.json 的 dict（或 AnalysisState 转成的 NovelDatabase.model_dump()）。
    :param book_id: 书籍 id，若为空则从 novel_db.book_id 取。
    :return: {"characters": n, "events": m}
    """
    bid = (book_id or "").strip() or (novel_db.get("book_id") or "")
    out: Dict[str, int] = {"characters": 0, "events": 0}
    if write_characters:
        out["characters"] = await write_characters_from_novel_db(session, novel_db, bid)
    if write_events:
        out["events"] = await write_events_from_novel_db(session, novel_db, bid)
    return out
