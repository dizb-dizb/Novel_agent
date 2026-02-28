# -*- coding: utf-8 -*-
"""
从分析产出的 novel_database 或原始书籍 JSON 构建逆向重构用的原著情节树 List[OriginalChapterNode]。

说明：分析流程中的「整合/重构」会把情节树合并，plot_tree 里 chapter_index 只有少量关键章
（如 1–9、173、253、332、394），因此按 novel_database 只能得到约 9～13 章。
需要 N 章（如 100/453）时，应使用原始书籍按章回退构建。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from backend.schemas.orchestrator_models import OriginalChapterNode
except ImportError:
    from schemas.orchestrator_models import OriginalChapterNode

logger = logging.getLogger(__name__)

# 当 novel_database 的 plot_tree 中不同 chapter_index 数量少于该阈值时，建议用原始书按章构建
MIN_CHAPTERS_FROM_PLOT_TREE = 20


def build_original_plot_tree_from_novel_db(
    novel_database: Any,
    *,
    max_chapters: Optional[int] = None,
    chapter_word_counts: Optional[Dict[int, int]] = None,
) -> List[OriginalChapterNode]:
    """
    从 novel_database（dict 或路径）构建按章节序排列的 OriginalChapterNode 列表。
    :param novel_database: 已加载的 novel_database 字典，或 novel_database.json 的 Path/str
    :param max_chapters: 最多取前 N 章，None 表示全部
    :param chapter_word_counts: 可选，章节号 -> 字数，用于 original_word_count
    :return: List[OriginalChapterNode]，按 chapter_number 升序
    """
    if isinstance(novel_database, (Path, str)):
        path = Path(novel_database)
        if not path.is_file():
            logger.warning("novel_database 文件不存在: %s", path)
            return []
        try:
            novel_database = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("读取 novel_database 失败: %s", e)
            return []

    if not isinstance(novel_database, dict):
        return []

    plot_tree = novel_database.get("plot_tree") or {}
    if not plot_tree:
        logger.warning("novel_database 中 plot_tree 为空")
        return []

    # 按 chapter_index 分组，收集 summary 作为 original_events
    by_chapter: Dict[int, List[Dict[str, Any]]] = {}
    for nid, node in plot_tree.items():
        if not isinstance(node, dict):
            continue
        ch = node.get("chapter_index")
        if ch is None:
            continue
        try:
            ch = int(ch)
        except (TypeError, ValueError):
            continue
        if ch not in by_chapter:
            by_chapter[ch] = []
        by_chapter[ch].append(node)

    # 每章内按 parent 顺序或保持列表顺序，汇总为 OriginalChapterNode
    nodes: List[OriginalChapterNode] = []
    for ch in sorted(by_chapter.keys()):
        if max_chapters is not None and ch > max_chapters:
            break
        items = by_chapter[ch]
        events = []
        goal = ""
        for n in items:
            s = (n.get("summary") or "").strip()
            if s:
                events.append(s)
                if not goal:
                    goal = s[:80] + ("…" if len(s) > 80 else "")
        word_count = None
        if chapter_word_counts and ch in chapter_word_counts:
            word_count = chapter_word_counts[ch]
        nodes.append(
            OriginalChapterNode(
                chapter_number=ch,
                original_pov="",
                original_events=events,
                original_goal=goal or f"第{ch}章情节",
                original_word_count=word_count,
            )
        )

    return nodes


def get_chapter_word_counts_from_raw_book(raw_book_path: Path) -> Dict[int, int]:
    """从原始书籍 JSON（含 chapters[].content）统计每章字数，返回 chapter_index(1-based) -> 字数。"""
    if not raw_book_path.is_file():
        return {}
    try:
        data = json.loads(raw_book_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    chapters = data.get("chapters") or []
    out = {}
    for i, ch in enumerate(chapters):
        if not isinstance(ch, dict):
            continue
        content = ch.get("content") or ch.get("text") or ""
        out[i + 1] = len(str(content).strip())
    return out


def build_original_plot_tree_from_raw_book(
    raw_book_path: Path,
    *,
    max_chapters: Optional[int] = None,
) -> List[OriginalChapterNode]:
    """
    从原始书籍 JSON 按章构建情节树：每章一个 OriginalChapterNode，保证 1..N 章连续。
    当 novel_database 的 plot_tree 因「整合/重构」只有少量关键章时，用本函数得到全书 N 章。
    """
    if not raw_book_path.is_file():
        logger.warning("原始书籍文件不存在: %s", raw_book_path)
        return []
    try:
        data = json.loads(raw_book_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("读取原始书籍失败: %s", e)
        return []
    chapters = data.get("chapters") or []
    if not chapters:
        return []
    n = len(chapters) if max_chapters is None else min(len(chapters), max_chapters)
    nodes: List[OriginalChapterNode] = []
    for i in range(n):
        ch = chapters[i] if isinstance(chapters[i], dict) else {}
        title = (ch.get("chapter_title") or ch.get("title") or "").strip() or f"第{i+1}章"
        content = (ch.get("content") or ch.get("text") or "").strip()
        summary = (content[:300] + "…") if len(content) > 300 else content
        if not summary:
            summary = title
        word_count = len(content)
        nodes.append(
            OriginalChapterNode(
                chapter_number=i + 1,
                original_pov="",
                original_events=[summary],
                original_goal=title,
                original_word_count=word_count if word_count else None,
            )
        )
    return nodes


def build_original_plot_tree(
    novel_database: Any,
    raw_book_path: Optional[Path] = None,
    *,
    max_chapters: Optional[int] = None,
    prefer_full_chapter_list: bool = True,
) -> List[OriginalChapterNode]:
    """
    优先从 novel_database 的 plot_tree 构建；若得到的章数过少（整合导致只有关键章），
    则改用原始书籍按章构建，保证 N 章连续（prefer_full_chapter_list=True 时）。
    """
    chapter_word_counts = None
    if raw_book_path and raw_book_path.is_file():
        chapter_word_counts = get_chapter_word_counts_from_raw_book(raw_book_path)
    from_db = build_original_plot_tree_from_novel_db(
        novel_database,
        max_chapters=max_chapters,
        chapter_word_counts=chapter_word_counts,
    )
    target = (max_chapters or 0) if max_chapters else (len(chapter_word_counts) if chapter_word_counts else 0)
    if prefer_full_chapter_list and raw_book_path and raw_book_path.is_file():
        need_count = max_chapters or len(chapter_word_counts) or 9999
        if len(from_db) < min(need_count, MIN_CHAPTERS_FROM_PLOT_TREE):
            logger.info(
                "novel_database 情节树仅 %s 章（整合后关键章），改用原始书籍按章构建共 %s 章",
                len(from_db),
                need_count if max_chapters else "全书",
            )
            return build_original_plot_tree_from_raw_book(raw_book_path, max_chapters=max_chapters)
    return from_db
