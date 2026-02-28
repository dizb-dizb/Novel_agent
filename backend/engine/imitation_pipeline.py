# -*- coding: utf-8 -*-
"""
仿写前置流水线：先执行分析形成四大知识库 + 文笔指纹，再供仿写循环使用。
确保「先分析、后仿写」，产出类似作者自己重写一篇的效果。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Tuple

try:
    from backend.schemas.orchestrator_models import StyleGuide
except ImportError:
    from schemas.orchestrator_models import StyleGuide

logger = logging.getLogger(__name__)


def style_fingerprint_to_style_guide(
    fp: Any,
    reference_book_name: Optional[str] = None,
) -> StyleGuide:
    """将分析产出的文笔指纹（style_fingerprint）转为仿写用的 StyleGuide。"""
    title = reference_book_name or (getattr(fp, "title", None) or getattr(fp, "book_id", "") or "参考书")
    # 词汇/句式：从 keyword_ratios 或 representative_descriptions 提炼
    vocab = []
    if getattr(fp, "keyword_ratios", None) and isinstance(fp.keyword_ratios, dict):
        vocab = list(fp.keyword_ratios.keys())[:20]
    if getattr(fp, "representative_descriptions", None) and isinstance(fp.representative_descriptions, list):
        for d in fp.representative_descriptions[:3]:
            if isinstance(d, str) and len(d) > 10:
                vocab.append(d[:80] + "…" if len(d) > 80 else d)
    pacing = ""
    if getattr(fp, "writing_habits", None) and fp.writing_habits:
        pacing = str(fp.writing_habits).strip()
    if getattr(fp, "sentence_style", None) and fp.sentence_style:
        pacing = (pacing + "；句式：" + str(fp.sentence_style).strip()).strip("；")
    dialogue = ""
    if getattr(fp, "rhetoric_notes", None) and fp.rhetoric_notes:
        dialogue = str(fp.rhetoric_notes).strip()
    if getattr(fp, "character_speech_samples", None) and isinstance(fp.character_speech_samples, list):
        parts = [f"{x.get('role', '')}: {x.get('sample', '')[:60]}" for x in fp.character_speech_samples if isinstance(x, dict)][:4]
        if parts:
            dialogue = (dialogue + " 角色口吻示例：" + " | ".join(parts)).strip()
    avg_len = None
    if getattr(fp, "avg_chapter_length", None) is not None:
        try:
            avg_len = float(fp.avg_chapter_length)
        except (TypeError, ValueError):
            pass
    return StyleGuide(
        reference_book_name=title,
        vocabulary_features=vocab[:25],
        pacing_rules=pacing,
        dialogue_style=dialogue,
        avg_chapter_length=avg_len,
    )


def load_style_guide_from_fingerprint_file(fp_path: Path) -> Optional[StyleGuide]:
    """从 style_fingerprint.json 加载并转为 StyleGuide。"""
    if not fp_path.is_file():
        return None
    try:
        data = json.loads(fp_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("读取文笔指纹失败 %s: %s", fp_path, e)
        return None
    try:
        from src.librarian.style_store import StyleFingerprint
        fp = StyleFingerprint.from_dict(data)
        return style_fingerprint_to_style_guide(fp)
    except ImportError:
        # 无 src 时用简易 dict 构造
        avg_len = data.get("avg_chapter_length")
        if avg_len is not None:
            try:
                avg_len = float(avg_len)
            except (TypeError, ValueError):
                avg_len = None
        return StyleGuide(
            reference_book_name=data.get("title") or data.get("book_id") or "参考书",
            vocabulary_features=[],
            pacing_rules=(data.get("writing_habits") or "") + " " + (data.get("sentence_style") or ""),
            dialogue_style=(data.get("rhetoric_notes") or ""),
            avg_chapter_length=avg_len,
        )


async def ensure_knowledge_bases_and_style(
    book_id: str,
    *,
    raw_dir: Optional[Path] = None,
    cards_dir: Optional[Path] = None,
    force_analyze: bool = False,
    max_chapters: Optional[int] = None,
    write_db: bool = True,
) -> Tuple[StyleGuide, Path]:
    """
    先执行分析形成四大知识库（角色、设定、情节、关系）并生成文笔指纹，再加载 StyleGuide。
    - 若 data/cards/{book_id}/novel_database.json 不存在或 force_analyze=True，则执行完整 analyze + 写库 + 文笔指纹。
    - 返回 (StyleGuide, novel_database.json 所在目录的 Path)，供仿写循环使用。
    """
    root = Path(__file__).resolve().parents[2]
    raw_dir = raw_dir or root / "data" / "raw"
    cards_dir = cards_dir or root / "data" / "cards"
    book_raw = raw_dir / book_id
    json_path = book_raw / f"{book_id}.json"
    out_book = cards_dir / book_id
    novel_db_path = out_book / "novel_database.json"
    fp_path = out_book / "style_fingerprint.json"

    if force_analyze or not novel_db_path.is_file():
        if not json_path.is_file():
            raise FileNotFoundError(
                f"未找到参考书原文: {json_path}。请先爬取该书或放置 {book_id}.json 到 data/raw/{book_id}/ 后再运行。"
            )
        logger.info("执行分析流水线: book_id=%s → 四大知识库 + 文笔指纹", book_id)
        try:
            from src.analyzer import build_novel_database, run_full_pipeline, save_state
        except ImportError as e:
            raise RuntimeError(f"请从项目根目录运行并确保 src.analyzer 可导入: {e}") from e
        data = json.loads(json_path.read_text(encoding="utf-8"))
        chapters = data.get("chapters") or []
        title = data.get("title") or book_id
        if max_chapters is not None:
            chapters = chapters[:max_chapters]
        if not chapters:
            raise ValueError(f"书籍无章节可分析: {json_path}")

        def _run():
            return run_full_pipeline(
                book_id=book_id,
                title=title,
                chapters=chapters,
                total_chapter_count=len(data.get("chapters") or []),
                refine_every_n_windows=1,
                use_concurrent_extraction=True,
                max_concurrent_workers=4,
            )

        import asyncio
        state = await asyncio.to_thread(_run)
        out_book.mkdir(parents=True, exist_ok=True)
        state_path = out_book / "analysis_state.json"
        save_state(state, state_path)
        novel_db = build_novel_database(state)
        novel_db_path.parent.mkdir(parents=True, exist_ok=True)
        novel_db_path.write_text(novel_db.model_dump_json(exclude_none=False, indent=2), encoding="utf-8")
        try:
            from src.librarian.style_store import build_style_fingerprint_library
            build_style_fingerprint_library(book_id, title, json_path, fp_path)
        except Exception as e:
            logger.warning("构建文笔指纹失败，将使用空风格: %s", e)
        if write_db:
            try:
                from src.analyzer.write_to_db import write_novel_database_to_backend
                result = write_novel_database_to_backend(book_id, novel_db_path)
                if result.get("ok"):
                    logger.info("已写入后端四大知识库: %s", result.get("written"))
                else:
                    logger.warning("写入后端失败: %s", result.get("error"))
            except Exception as e:
                logger.warning("写入后端失败: %s", e)
    else:
        logger.info("使用已有知识库与文笔指纹: %s", out_book)

    style_guide = load_style_guide_from_fingerprint_file(fp_path)
    if not style_guide:
        style_guide = StyleGuide(
            reference_book_name=book_id,
            vocabulary_features=[],
            pacing_rules="",
            dialogue_style="",
        )
    return style_guide, out_book
