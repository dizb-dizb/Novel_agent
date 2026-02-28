# -*- coding: utf-8 -*-
"""
总装流水线：带分析反馈的滚动生成大循环。
串联风格、逻辑主编、编写模块与分析建库，阶段 C 失败则暂停，防止逻辑崩塌。
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

try:
    from backend.schemas.orchestrator_models import (
        BookState,
        ChapterDesign,
        ChapterMeta,
        MutationPremise,
        OriginalChapterNode,
        ReconstructedOutline,
        StyleGuide,
    )
    from backend.schemas.writing_schemas import WriteRequest, WriteResponse
except ImportError:
    from schemas.orchestrator_models import (
        BookState,
        ChapterDesign,
        ChapterMeta,
        MutationPremise,
        OriginalChapterNode,
        ReconstructedOutline,
        StyleGuide,
    )
    from schemas.writing_schemas import WriteRequest, WriteResponse

logger = logging.getLogger(__name__)

# ---------- AnalysisAgent 接口 ----------


class AnalysisAgent:
    """
    分析 Agent：将新生成章节的事实与人物状态变动写入数据库（史官角色）。
    若阶段 C 更新数据库失败，流水线必须暂停。
    """

    async def extract_and_update_db(
        self,
        book_id: str,
        chapter_number: int,
        chapter_text: str,
        *,
        session=None,
    ) -> None:
        """
        对新章节正文进行抽取并更新四大数据库（角色、设定、关系、情节）。
        :param session: 可选，AsyncSession；若不传则内部创建并 commit。
        :raises: 任意异常时调用方应暂停流水线。
        """
        raise NotImplementedError(
            "请实现 AnalysisAgent.extract_and_update_db：可对接现有分析流水线 + analysis_import_service.write_novel_database_to_db"
        )


# ---------- 默认分析实现（可替换） ----------


class DefaultAnalysisAgent(AnalysisAgent):
    """
    默认实现：将本章正文转为最小 novel_database 并写入 DB。
    生产环境可替换为调用完整分析流水线（如 src.analyzer）。
    """

    async def extract_and_update_db(
        self,
        book_id: str,
        chapter_number: int,
        chapter_text: str,
        *,
        session=None,
    ) -> None:
        try:
            from backend.database import AsyncSessionLocal
            from backend.services.analysis_import_service import write_novel_database_to_db
        except ImportError:
            from database import AsyncSessionLocal
            from services.analysis_import_service import write_novel_database_to_db

        # 最小 novel_database：仅记录本章情节节点，便于后续扩展为完整抽取
        novel_db = {
            "book_id": book_id,
            "plot_tree": {
                f"ch{chapter_number}": {
                    "id": f"ch{chapter_number}",
                    "parent_id": None,
                    "type": "chapter",
                    "summary": (chapter_text or "")[:500].strip() or f"第{chapter_number}章",
                    "chapter_index": chapter_number,
                    "involved_characters": None,
                    "outcomes": None,
                },
            },
            "cards": [],
            "entities_by_type": {},
        }
        if session is not None:
            await write_novel_database_to_db(session, novel_db, book_id)
            return
        async with AsyncSessionLocal() as sess:
            await write_novel_database_to_db(sess, novel_db, book_id)
            await sess.commit()


# ---------- 四大知识库 RAG 组装（供仿写时严格贴合已分析情节与角色） ----------


async def assemble_rag_from_backend(request: "WriteRequest", book_id: str) -> str:
    """
    从后端四大知识库（角色、情节、设定、关系）组装 RAG 上下文 YAML。
    仿写时必须使用本函数，使正文与已分析的知识库一致，呈现「作者重写」感。
    """
    try:
        from backend.database import AsyncSessionLocal
        from backend.models.character import Character
        from backend.models.plot import Event
        from sqlalchemy import select
    except ImportError:
        from database import AsyncSessionLocal
        from models.character import Character
        from models.plot import Event
        from sqlalchemy import select

    lines = ["# 四大知识库 RAG 上下文（来自分析结果）", ""]
    async with AsyncSessionLocal() as session:
        # 角色
        chars = await session.execute(select(Character).where(Character.book_id == book_id))
        characters = list(chars.scalars().all() or [])
        focus = set((request.focus_character_ids or [])[:20])
        lines.append("[角色/人物]")
        for c in characters[:50]:
            name = (c.name or "").strip()
            if not name:
                continue
            mark = " (本章视角)" if name in focus else ""
            lines.append(f"  - 姓名: {name}{mark}")
            if c.personality_profile:
                lines.append(f"    性格: {c.personality_profile[:200]}")
            if c.speaking_style:
                lines.append(f"    说话习惯: {c.speaking_style[:150]}")
        if not characters:
            lines.append("  - （暂无，请勿捏造未出现角色）")
        lines.append("")

        # 情节树（前情提要）
        events = await session.execute(
            select(Event).where(Event.book_id == book_id).order_by(Event.sequence_order)
        )
        plot_list = list(events.scalars().all() or [])
        lines.append("[前情提要/情节树]")
        for e in plot_list[-30:]:  # 最近 30 条
            s = (e.summary or "").strip()
            if s:
                lines.append(f"  - {s[:150]}")
        if not plot_list:
            lines.append("  - （暂无，请按主线自然发展）")
        lines.append("")

        # 设定、关系（当前后端若暂无独立表，可后续扩展）
        lines.append("[设定与关系]")
        lines.append("  - 由设定库与关系图谱提供（若已写入后端则此处可扩展检索）")
    return "\n".join(lines)


# ---------- AutoNovelEngine ----------


class AutoNovelEngine:
    """
    自循环仿写引擎：风格加载 → 每章「逻辑审查 → 受控编写 → 分析固化 → 状态推进」。
    阶段 C（分析入库）失败时立即暂停并抛出，不进入下一章。
    """

    def __init__(
        self,
        *,
        logic_master: "LogicMasterAgent",
        writing_agent: "Any",  # WritingAgent
        analysis_agent: AnalysisAgent,
        get_db_snapshot: Optional[Callable[[str], Awaitable[Dict[str, Any]]]] = None,
    ) -> None:
        self.logic_master = logic_master
        self.writing_agent = writing_agent
        self.analysis_agent = analysis_agent
        self.get_db_snapshot = get_db_snapshot or self._default_db_snapshot

    async def _default_db_snapshot(self, book_id: str) -> Dict[str, Any]:
        """默认快照：从 SQLite 拉取角色与情节，供 LogicMaster 使用。"""
        try:
            from backend.database import AsyncSessionLocal
            from backend.models.character import Character
            from backend.models.plot import Event
            from sqlalchemy import select
        except ImportError:
            from database import AsyncSessionLocal
            from models.character import Character
            from models.plot import Event
            from sqlalchemy import select

        async with AsyncSessionLocal() as session:
            chars = await session.execute(select(Character).where(Character.book_id == book_id))
            characters = [
                {"id": c.id, "name": c.name, "personality_profile": c.personality_profile}
                for c in (chars.scalars().all() or [])
            ]
            events = await session.execute(select(Event).where(Event.book_id == book_id).order_by(Event.sequence_order))
            plot_events = [
                {"id": e.id, "summary": e.summary, "level": e.level.value if e.level else None}
                for e in (events.scalars().all() or [])
            ]
        return {"characters": characters, "plot_events": plot_events, "settings": [], "relations": []}

    def _build_write_request(
        self,
        design: ChapterDesign,
        style_guide: StyleGuide,
        *,
        render_only: bool = False,
        target_chapter_length: Optional[int] = None,
    ) -> WriteRequest:
        """根据单章设计图与风格指南构造 WritingAgent 的 WriteRequest。篇幅优先用 target_chapter_length，否则用 style_guide.avg_chapter_length。"""
        parts = []
        if design.required_events:
            parts.append("本章必须发生的事件：\n" + "\n".join(f"- {e}" for e in design.required_events))
        if design.logic_constraints:
            parts.append("逻辑约束（必须遵守）：\n" + "\n".join(f"- {c}" for c in design.logic_constraints))
        if style_guide.pacing_rules:
            parts.append("行文节奏要求：" + style_guide.pacing_rules)
        if style_guide.dialogue_style:
            parts.append("对话风格：" + style_guide.dialogue_style)
        if style_guide.vocabulary_features:
            parts.append("可参考的词汇/句式：" + "；".join(style_guide.vocabulary_features[:15]))
        # 篇幅：满足原文指纹或本章指定字数
        word_target = target_chapter_length
        if word_target is None and getattr(style_guide, "avg_chapter_length", None) not in (None, 0):
            try:
                word_target = int(style_guide.avg_chapter_length)
            except (TypeError, ValueError):
                pass
        if word_target and word_target > 0:
            parts.append(f"【篇幅】本章正文字数需约 {word_target} 字，与参考书篇幅一致，不得明显偏短或偏长。")
        if render_only:
            parts.append("【严格】仅按上述细纲与事件渲染正文，禁止发散、增删关键情节或改变因果。")
        user_instruction = "\n\n".join(parts) if parts else "按当前主线自然续写本章，保持逻辑一致。"
        focus_ids = [design.pov_character] if design.pov_character else []
        return WriteRequest(user_instruction=user_instruction, focus_character_ids=focus_ids, location_id=None)

    async def run_imitation_loop(
        self,
        book_id: str,
        target_chapters: int,
        *,
        style_guide: Optional[StyleGuide] = None,
        initial_state: Optional[BookState] = None,
    ) -> BookState:
        """
        主循环：按章节递增，执行 逻辑审查 → 受控编写 → 分析固化 → 状态推进。
        风格在循环外传入；若未传 style_guide 则使用空风格（仅逻辑与设计图约束）。
        阶段 C 失败则暂停并抛出，返回当前 BookState。
        """
        style_guide = style_guide or StyleGuide(reference_book_name="", vocabulary_features=[], pacing_rules="", dialogue_style="")
        state = initial_state or BookState(current_chapter=0, main_plot_goal="")
        logger.info("run_imitation_loop 启动 book_id=%s target_chapters=%s", book_id, target_chapters)

        while state.current_chapter < target_chapters:
            next_chapter = state.current_chapter + 1
            logger.info("======== 第 %s 章 ========", next_chapter)

            # ---------- 阶段 A：逻辑审查与设计 ----------
            try:
                db_snapshot = await self.get_db_snapshot(book_id)
            except Exception as e:
                logger.exception("阶段 A 前获取 db_snapshot 失败")
                raise
            try:
                design = await self.logic_master.review_and_design(state, db_snapshot)
                design.chapter_number = next_chapter
            except Exception as e:
                logger.exception("阶段 A review_and_design 失败")
                raise

            # ---------- 阶段 B：受控编写（严格采用 StyleGuide 文笔 + 设计图逻辑 + 四大知识库 RAG） ----------
            try:
                write_request = self._build_write_request(design, style_guide)
                response: WriteResponse = await self.writing_agent.generate_chapter(
                    write_request, book_id=book_id, style_guide=style_guide
                )
            except Exception as e:
                logger.exception("阶段 B generate_chapter 失败")
                raise

            # ---------- 阶段 C：分析与知识固化（失败则暂停） ----------
            try:
                await self.analysis_agent.extract_and_update_db(
                    book_id, next_chapter, response.draft_content
                )
            except Exception as e:
                logger.exception("阶段 C extract_and_update_db 失败，流水线暂停")
                raise RuntimeError(f"阶段 C 更新数据库失败，已暂停：{e!s}") from e

            # ---------- 阶段 D：状态推进 ----------
            state = BookState(
                current_chapter=next_chapter,
                main_plot_goal=state.main_plot_goal,
                note=state.note,
            )
            logger.info("第 %s 章完成，当前进度 %s/%s", next_chapter, state.current_chapter, target_chapters)

        return state

    async def run_reconstruction_loop(
        self,
        book_id: str,
        original_plot_tree: List[Any],
        mutation_premise: MutationPremise,
        *,
        style_guide: Optional[StyleGuide] = None,
    ) -> BookState:
        """
        逆向重构主循环：遍历原著情节树，逐章「逻辑适配 → 受控渲染 → 分析固化 → 状态推进」。
        Phase 0 假设：循环外已提供完整 List[OriginalChapterNode]、MutationPremise、StyleGuide。
        """
        style_guide = style_guide or StyleGuide(
            reference_book_name="", vocabulary_features=[], pacing_rules="", dialogue_style=""
        )
        if not original_plot_tree:
            logger.warning("原著情节树为空，直接返回")
            return BookState(current_chapter=0, main_plot_goal="")

        tree: List[OriginalChapterNode] = [
            n if isinstance(n, OriginalChapterNode) else OriginalChapterNode(**n)
            for n in original_plot_tree
        ]
        logger.info(
            "run_reconstruction_loop 启动 book_id=%s 原著章数=%s",
            book_id,
            len(tree),
        )

        for idx, original_chapter in enumerate(tree):
            chapter_number = idx + 1
            logger.info("======== 第 %s 章（原著节点 %s）======== ", chapter_number, original_chapter.chapter_number)

            # ---------- 阶段 A：逻辑适配 ----------
            try:
                db_snapshot = await self.get_db_snapshot(book_id)
            except Exception as e:
                logger.exception("阶段 A 前获取 db_snapshot 失败")
                raise
            try:
                design = await self.logic_master.adapt_and_design(
                    original_chapter, mutation_premise, db_snapshot
                )
                design.chapter_number = chapter_number
            except Exception as e:
                logger.exception("阶段 A adapt_and_design 失败")
                raise

            # ---------- 阶段 B：受控渲染（仅做渲染，禁止发散；篇幅满足原文指纹或本章字数） ----------
            try:
                target_len = getattr(original_chapter, "original_word_count", None) if original_chapter else None
                write_request = self._build_write_request(
                    design, style_guide, render_only=True, target_chapter_length=target_len
                )
                response: WriteResponse = await self.writing_agent.generate_chapter(
                    write_request, book_id=book_id, style_guide=style_guide
                )
            except Exception as e:
                logger.exception("阶段 B generate_chapter 失败")
                raise

            # ---------- 阶段 C：分析与固化 ----------
            try:
                await self.analysis_agent.extract_and_update_db(
                    book_id, chapter_number, response.draft_content
                )
            except Exception as e:
                logger.exception("阶段 C extract_and_update_db 失败，流水线暂停")
                raise RuntimeError(f"阶段 C 更新数据库失败，已暂停：{e!s}") from e

            # ---------- 阶段 D：状态推进与日志 ----------
            orig_goal = original_chapter.original_goal or "（无）"
            new_events = design.required_events[:3]
            new_summary = "；".join(new_events) if new_events else "（无）"
            logger.info(
                "第 %s 章重构渲染完成，原著节点 [%s] 已适配为 -> [%s]",
                chapter_number,
                orig_goal,
                new_summary,
            )

        return BookState(current_chapter=len(tree), main_plot_goal="")

    async def run_render_from_outline(
        self,
        book_id: str,
        outline: Any,
        *,
        style_guide: Optional[StyleGuide] = None,
    ) -> BookState:
        """
        按已落盘的知识框架逐章实现文本：仅编写 + 分析固化，不做逻辑适配。
        outline 为 ReconstructedOutline 或 reconstructed_outline.json 路径。
        """
        if isinstance(outline, (Path, str)):
            path = Path(outline)
            if not path.is_file():
                raise FileNotFoundError(f"知识框架文件不存在: {path}")
            data = json.loads(path.read_text(encoding="utf-8"))
            outline = ReconstructedOutline.model_validate(data)
        if not isinstance(outline, ReconstructedOutline):
            outline = ReconstructedOutline.model_validate(outline)

        style_guide = style_guide or StyleGuide(
            reference_book_name="", vocabulary_features=[], pacing_rules="", dialogue_style=""
        )
        designs = outline.designs
        meta_list = outline.chapter_meta or []
        if len(meta_list) < len(designs):
            meta_list = meta_list + [ChapterMeta(chapter_number=i + 1) for i in range(len(meta_list), len(designs))]

        logger.info("run_render_from_outline 启动 book_id=%s 共 %s 章（仅文本实现）", book_id, len(designs))

        for idx, design in enumerate(designs):
            chapter_number = idx + 1
            meta = meta_list[idx] if idx < len(meta_list) else ChapterMeta(chapter_number=chapter_number)
            logger.info("======== 第 %s 章 文本实现 ========", chapter_number)

            target_len = getattr(meta, "original_word_count", None) if meta else None
            write_request = self._build_write_request(
                design, style_guide, render_only=True, target_chapter_length=target_len
            )
            response: WriteResponse = await self.writing_agent.generate_chapter(
                write_request, book_id=book_id, style_guide=style_guide
            )
            await self.analysis_agent.extract_and_update_db(
                book_id, chapter_number, response.draft_content
            )
            logger.info(
                "第 %s 章文本实现完成，框架目标 [%s] -> 已渲染并固化",
                chapter_number,
                (meta.original_goal[:40] + "…") if (meta and meta.original_goal) else "（无）",
            )

        return BookState(current_chapter=len(designs), main_plot_goal="")

    # ---------- 并发流水线：生产者-消费者 ----------

    async def run_reconstruction_loop_concurrent(
        self,
        book_id: str,
        original_plot_tree: List[Any],
        mutation_premise: MutationPremise,
        *,
        style_guide: Optional[StyleGuide] = None,
    ) -> BookState:
        """
        基于 asyncio.Queue 的并发流水线：大脑(设计) → 打字机(渲染) → 后台(分析入库)。
        使用 design_queue / writing_queue，三阶段并发执行；DB 快照可适度乐观，不强制等待上一章分析完毕。
        """
        style_guide = style_guide or StyleGuide(
            reference_book_name="", vocabulary_features=[], pacing_rules="", dialogue_style=""
        )
        if not original_plot_tree:
            logger.warning("原著情节树为空，直接返回")
            return BookState(current_chapter=0, main_plot_goal="")

        tree: List[OriginalChapterNode] = [
            n if isinstance(n, OriginalChapterNode) else OriginalChapterNode(**n)
            for n in original_plot_tree
        ]
        design_queue: asyncio.Queue = asyncio.Queue()
        writing_queue: asyncio.Queue = asyncio.Queue()

        async def producer_logic() -> None:
            """生产者：遍历情节树，生成 ChapterDesign 并放入 design_queue。"""
            db_snapshot = await self.get_db_snapshot(book_id)
            for idx, original_chapter in enumerate(tree):
                chapter_number = idx + 1
                logger.info("[大脑] 正在设计第 %s 章...", chapter_number)
                try:
                    design = await self.logic_master.adapt_and_design(
                        original_chapter, mutation_premise, db_snapshot
                    )
                    design.chapter_number = chapter_number
                    design.adapted_from_chapter = design.adapted_from_chapter or original_chapter.chapter_number
                    await design_queue.put((chapter_number, design, getattr(original_chapter, "original_word_count", None)))
                except Exception as e:
                    logger.exception("[大脑] 第 %s 章设计失败", chapter_number)
                    await design_queue.put((chapter_number, None, None))
            await design_queue.put(None)

        async def consumer_writing() -> None:
            """消费者：从 design_queue 取设计，渲染正文，放入 writing_queue。"""
            while True:
                item = await design_queue.get()
                if item is None:
                    design_queue.task_done()
                    await writing_queue.put(None)
                    return
                chapter_number, design, target_len = item
                design_queue.task_done()
                if design is None:
                    continue
                logger.info("[打字机] 正在渲染第 %s 章...", chapter_number)
                try:
                    write_request = self._build_write_request(
                        design, style_guide, render_only=True, target_chapter_length=target_len
                    )
                    response: WriteResponse = await self.writing_agent.generate_chapter(
                        write_request, book_id=book_id, style_guide=style_guide
                    )
                    await writing_queue.put((chapter_number, response.draft_content))
                except Exception as e:
                    logger.exception("[打字机] 第 %s 章渲染失败: %s", chapter_number, e)

        async def background_analysis() -> None:
            """后台：从 writing_queue 取正文，分析并写入 DB。"""
            while True:
                item = await writing_queue.get()
                if item is None:
                    writing_queue.task_done()
                    return
                chapter_number, draft_content = item
                writing_queue.task_done()
                logger.info("[后台] 第 %s 章知识入库中...", chapter_number)
                try:
                    await self.analysis_agent.extract_and_update_db(
                        book_id, chapter_number, draft_content
                    )
                    logger.info("[后台] 第 %s 章知识入库完成", chapter_number)
                except Exception as e:
                    logger.exception("[后台] 第 %s 章入库失败: %s", chapter_number, e)

        logger.info(
            "run_reconstruction_loop_concurrent 启动 book_id=%s 原著章数=%s（大脑/打字机/后台并发）",
            book_id,
            len(tree),
        )
        await asyncio.gather(producer_logic(), consumer_writing(), background_analysis())
        return BookState(current_chapter=len(tree), main_plot_goal="")

    async def run_render_from_outline_concurrent(
        self,
        book_id: str,
        outline: Any,
        *,
        style_guide: Optional[StyleGuide] = None,
        render_workers: int = 3,
    ) -> BookState:
        """
        按已落盘的知识框架并发实现文本：设计队列为 outline 的 designs，
        多个打字机 worker 并行渲染（每出一个设计就由空闲 worker 接走）→ 后台入库。
        render_workers：并行渲染的 worker 数，默认 3。
        """
        if isinstance(outline, (Path, str)):
            path = Path(outline)
            if not path.is_file():
                raise FileNotFoundError(f"知识框架文件不存在: {path}")
            data = json.loads(path.read_text(encoding="utf-8"))
            outline = ReconstructedOutline.model_validate(data)
        if not isinstance(outline, ReconstructedOutline):
            outline = ReconstructedOutline.model_validate(outline)

        style_guide = style_guide or StyleGuide(
            reference_book_name="", vocabulary_features=[], pacing_rules="", dialogue_style=""
        )
        designs = outline.designs
        meta_list = outline.chapter_meta or []
        if len(meta_list) < len(designs):
            meta_list = meta_list + [ChapterMeta(chapter_number=i + 1) for i in range(len(meta_list), len(designs))]

        design_queue: asyncio.Queue = asyncio.Queue()
        writing_queue: asyncio.Queue = asyncio.Queue()
        workers_done_lock = asyncio.Lock()
        remaining_workers = render_workers

        async def producer_outline() -> None:
            for idx, design in enumerate(designs):
                ch = idx + 1
                meta = meta_list[idx] if idx < len(meta_list) else ChapterMeta(chapter_number=ch)
                target_len = getattr(meta, "original_word_count", None)
                await design_queue.put((ch, design, target_len))
            await design_queue.put(None)

        async def consumer_writing_worker(_worker_id: int) -> None:
            nonlocal remaining_workers
            while True:
                item = await design_queue.get()
                if item is None:
                    design_queue.task_done()
                    await design_queue.put(None)
                    async with workers_done_lock:
                        remaining_workers -= 1
                        if remaining_workers == 0:
                            await writing_queue.put(None)
                    return
                chapter_number, design, target_len = item
                design_queue.task_done()
                logger.info("[打字机-%s] 正在渲染第 %s 章...", _worker_id, chapter_number)
                try:
                    write_request = self._build_write_request(
                        design, style_guide, render_only=True, target_chapter_length=target_len
                    )
                    response: WriteResponse = await self.writing_agent.generate_chapter(
                        write_request, book_id=book_id, style_guide=style_guide
                    )
                    await writing_queue.put((chapter_number, response.draft_content))
                    logger.info("[打字机-%s] 第 %s 章渲染完成", _worker_id, chapter_number)
                except Exception as e:
                    logger.exception("[打字机-%s] 第 %s 章渲染失败: %s", _worker_id, chapter_number, e)

        async def background_analysis() -> None:
            while True:
                item = await writing_queue.get()
                if item is None:
                    writing_queue.task_done()
                    return
                chapter_number, draft_content = item
                writing_queue.task_done()
                logger.info("[后台] 第 %s 章知识入库中...", chapter_number)
                try:
                    await self.analysis_agent.extract_and_update_db(
                        book_id, chapter_number, draft_content
                    )
                    logger.info("[后台] 第 %s 章知识入库完成", chapter_number)
                except Exception as e:
                    logger.exception("[后台] 第 %s 章入库失败: %s", chapter_number, e)

        logger.info(
            "run_render_from_outline_concurrent 启动 book_id=%s 共 %s 章（%s 个打字机并行 + 后台入库）",
            book_id,
            len(designs),
            render_workers,
        )
        await asyncio.gather(
            producer_outline(),
            *[consumer_writing_worker(w) for w in range(1, render_workers + 1)],
            background_analysis(),
        )
        return BookState(current_chapter=len(designs), main_plot_goal="")
