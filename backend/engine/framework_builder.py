# -*- coding: utf-8 -*-
"""
整体搭建知识框架：从原著情节树 + 变异基调，一次性产出全书 N 章的适配细纲并落盘。
编写模块仅按此框架逐章实现文本，不再在写每章时做逻辑设计。
"""
from __future__ import annotations

import asyncio
import copy
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from backend.schemas.orchestrator_models import (
        ChapterDesign,
        ChapterMeta,
        MutationPremise,
        OriginalChapterNode,
        ReconstructedOutline,
    )
except ImportError:
    from schemas.orchestrator_models import (
        ChapterDesign,
        ChapterMeta,
        MutationPremise,
        OriginalChapterNode,
        ReconstructedOutline,
    )

logger = logging.getLogger(__name__)


def prune_db_snapshot_for_batch(
    snapshot: Dict[str, Any],
    original_nodes_batch: List[Any],
    *,
    max_characters: int = 80,
    max_plot_events: int = 50,
    max_relations: int = 30,
    max_settings: int = 20,
) -> Dict[str, Any]:
    """
    快照降维：只保留与本批原著章节强相关的数据，减少传给 LogicMaster 的 token。
    - 角色：优先保留在本批 original_pov / original_goal / original_events 中出现的，再补足前 N 个主 cast，总数不超过 max_characters。
    - 情节：保留最近 max_plot_events 条。
    - 关系：保留最近 max_relations 条（如「最近 3 条核心关系链」可设 3）。
    - 设定：保留前 max_settings 条。
    """
    if not snapshot:
        return snapshot
    pruned = copy.deepcopy(snapshot)

    # 本批文本：用于匹配「出场人物」
    batch_text_parts = []
    for n in original_nodes_batch:
        node = n if isinstance(n, OriginalChapterNode) else OriginalChapterNode(**n)
        batch_text_parts.append(node.original_pov or "")
        batch_text_parts.append(node.original_goal or "")
        batch_text_parts.extend(node.original_events or [])
    batch_text = " ".join(str(x) for x in batch_text_parts)
    # 本批原著中明确出现的人名（POV 等），用于补全快照，避免「视角人物不在 DB」误杀（分析可能只写入少量角色）
    names_from_batch = set()
    for n in original_nodes_batch:
        node = n if isinstance(n, OriginalChapterNode) else OriginalChapterNode(**n)
        if (node.original_pov or "").strip():
            names_from_batch.add((node.original_pov or "").strip())

    # 角色：多 key 兼容（characters / 角色 / 人物）
    for key in ("characters", "角色", "人物", "entities_by_type"):
        raw = pruned.get(key)
        if not isinstance(raw, list):
            raw = []
        names_to_keep = set()
        for c in raw:
            if isinstance(c, dict):
                name = (c.get("name") or c.get("id") or "").strip()
            elif isinstance(c, str):
                name = c.strip()
            else:
                continue
            if name and name in batch_text:
                names_to_keep.add(name)
        # 保留「本批出现」+ 前 15 个作为主 cast，再截断总数
        kept = []
        for c in raw:
            if isinstance(c, dict):
                name = (c.get("name") or c.get("id") or "").strip()
            elif isinstance(c, str):
                name = c.strip()
            else:
                continue
            if name and (name in names_to_keep or len(kept) < 15):
                kept.append(c)
            if len(kept) >= max_characters:
                break
        # 补全：本批原著 POV 等名字若不在 DB，也加入快照，避免逻辑主编校验报「视角人物不在列表」
        existing_names = {((c.get("name") or c.get("id") or "").strip()) for c in kept if isinstance(c, dict)}
        for name in names_from_batch:
            if name and name not in existing_names:
                kept.append({"name": name, "id": name})
                existing_names.add(name)
            if len(kept) >= max_characters:
                break
        pruned[key] = kept[:max_characters]

    # 情节：保留最近 N 条
    for key in ("plot_events", "plot_tree", "情节"):
        val = pruned.get(key)
        if isinstance(val, list):
            pruned[key] = val[-max_plot_events:] if len(val) > max_plot_events else val
        elif isinstance(val, dict):
            items = list(val.items())
            if len(items) > max_plot_events:
                pruned[key] = dict(items[-max_plot_events:])
            # else 保持原样

    # 关系：最近 N 条
    for key in ("relations", "关系"):
        val = pruned.get(key)
        if isinstance(val, list):
            pruned[key] = val[-max_relations:] if len(val) > max_relations else val

    # 设定：前 N 条
    for key in ("settings", "设定"):
        val = pruned.get(key)
        if isinstance(val, list):
            pruned[key] = val[:max_settings]

    return pruned


async def build_reconstructed_framework(
    book_id: str,
    original_plot_tree: List[Any],
    mutation_premise: MutationPremise,
    logic_master: Any,
    get_db_snapshot: Callable[[str], Any],
    *,
    outline_path: Optional[Path] = None,
    cards_dir: Optional[Path] = None,
    batch_size: int = 5,
) -> ReconstructedOutline:
    """
    整体搭建知识框架：对全书 N 章（original_plot_tree）在知识层面完成逻辑适配，
    产出 N 个 ChapterDesign，与 chapter_meta 一并落盘，供后续「仅文本实现」使用。
    按 batch_size 分批调用 review_and_design_batch，摊薄大模型推理耗时。
    使用同一份 db_snapshot（分析后的四大知识库）做全部分章适配，保证框架一致。
    """
    tree: List[OriginalChapterNode] = [
        n if isinstance(n, OriginalChapterNode) else OriginalChapterNode(**n)
        for n in original_plot_tree
    ]
    if not tree:
        raise ValueError("原著情节树为空，无法搭建知识框架")

    db_snapshot = await get_db_snapshot(book_id)
    designs: List[ChapterDesign] = []
    chapter_meta: List[ChapterMeta] = []

    batch_size = max(1, batch_size)
    total = len(tree)
    for start in range(0, total, batch_size):
        batch = tree[start : start + batch_size]
        batch_num = start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        logger.info(
            "知识框架 批量适配第 %s/%s 批（章 %s-%s，共 %s 章）",
            batch_num,
            total_batches,
            start + 1,
            start + len(batch),
            len(batch),
        )
        snapshot_to_use = prune_db_snapshot_for_batch(
            db_snapshot,
            batch,
            max_characters=80,
            max_plot_events=50,
            max_relations=30,
            max_settings=20,
        )
        # 校验使用注入本批人名的 snapshot_to_use，避免分析只写入少量角色时误报「视角人物不在列表」
        if len(batch) == 1:
            design = await logic_master.adapt_and_design(
                batch[0],
                mutation_premise,
                snapshot_to_use,
                validate_against_snapshot=snapshot_to_use,
            )
            design.chapter_number = batch[0].chapter_number
            design.adapted_from_chapter = design.adapted_from_chapter or batch[0].chapter_number
            batch_designs = [design]
        else:
            batch_designs = await logic_master.review_and_design_batch(
                batch,
                mutation_premise,
                snapshot_to_use,
                validate_against_snapshot=snapshot_to_use,
            )
        for i, (orig, design) in enumerate(zip(batch, batch_designs)):
            ch_num = start + i + 1
            design.chapter_number = ch_num
            design.adapted_from_chapter = design.adapted_from_chapter or orig.chapter_number
            designs.append(design)
            chapter_meta.append(
                ChapterMeta(
                    chapter_number=ch_num,
                    original_goal=orig.original_goal or "",
                    original_word_count=getattr(orig, "original_word_count", None),
                )
            )

    outline = ReconstructedOutline(
        book_id=book_id,
        total_chapters=len(designs),
        mutation_premise=mutation_premise,
        designs=designs,
        chapter_meta=chapter_meta,
    )

    if outline_path is None and cards_dir:
        outline_path = cards_dir / book_id / "reconstructed_outline.json"
    if outline_path:
        outline_path = Path(outline_path)
        outline_path.parent.mkdir(parents=True, exist_ok=True)
        outline_path.write_text(
            outline.model_dump_json(exclude_none=True, indent=2),
            encoding="utf-8",
        )
        logger.info("知识框架已落盘: %s（共 %s 章）", outline_path, outline.total_chapters)

    return outline


async def build_reconstructed_framework_streaming(
    book_id: str,
    original_plot_tree: List[Any],
    mutation_premise: MutationPremise,
    logic_master: Any,
    get_db_snapshot: Callable[[str], Any],
    design_queue: asyncio.Queue,
    *,
    outline_path: Optional[Path] = None,
    cards_dir: Optional[Path] = None,
    batch_size: int = 5,
) -> ReconstructedOutline:
    """
    流式搭建知识框架：每产出一批设计就放入 design_queue，供渲染端并发消费，实现「搭框架」与「渲染」流水线并行。
    同时仍在内存中收集完整 outline，结束时落盘。
    """
    tree: List[OriginalChapterNode] = [
        n if isinstance(n, OriginalChapterNode) else OriginalChapterNode(**n)
        for n in original_plot_tree
    ]
    if not tree:
        raise ValueError("原著情节树为空，无法搭建知识框架")

    db_snapshot = await get_db_snapshot(book_id)
    designs: List[ChapterDesign] = []
    chapter_meta: List[ChapterMeta] = []

    batch_size = max(1, batch_size)
    total = len(tree)
    for start in range(0, total, batch_size):
        batch = tree[start : start + batch_size]
        batch_num = start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        logger.info(
            "[框架] 批量适配第 %s/%s 批（章 %s-%s），产出后即入队供渲染",
            batch_num,
            total_batches,
            start + 1,
            start + len(batch),
            len(batch),
        )
        snapshot_to_use = prune_db_snapshot_for_batch(
            db_snapshot,
            batch,
            max_characters=80,
            max_plot_events=50,
            max_relations=30,
            max_settings=20,
        )
        if len(batch) == 1:
            design = await logic_master.adapt_and_design(
                batch[0],
                mutation_premise,
                snapshot_to_use,
                validate_against_snapshot=snapshot_to_use,
            )
            design.chapter_number = batch[0].chapter_number
            design.adapted_from_chapter = design.adapted_from_chapter or batch[0].chapter_number
            batch_designs = [design]
        else:
            batch_designs = await logic_master.review_and_design_batch(
                batch,
                mutation_premise,
                snapshot_to_use,
                validate_against_snapshot=snapshot_to_use,
            )
        for i, (orig, design) in enumerate(zip(batch, batch_designs)):
            ch_num = start + i + 1
            design.chapter_number = ch_num
            design.adapted_from_chapter = design.adapted_from_chapter or orig.chapter_number
            meta = ChapterMeta(
                chapter_number=ch_num,
                original_goal=orig.original_goal or "",
                original_word_count=getattr(orig, "original_word_count", None),
            )
            designs.append(design)
            chapter_meta.append(meta)
            await design_queue.put((ch_num, design, meta))

    await design_queue.put(None)

    outline = ReconstructedOutline(
        book_id=book_id,
        total_chapters=len(designs),
        mutation_premise=mutation_premise,
        designs=designs,
        chapter_meta=chapter_meta,
    )
    if outline_path is None and cards_dir:
        outline_path = cards_dir / book_id / "reconstructed_outline.json"
    if outline_path:
        outline_path = Path(outline_path)
        outline_path.parent.mkdir(parents=True, exist_ok=True)
        outline_path.write_text(
            outline.model_dump_json(exclude_none=True, indent=2),
            encoding="utf-8",
        )
        logger.info("知识框架已落盘: %s（共 %s 章）", outline_path, outline.total_chapters)
    return outline
