# -*- coding: utf-8 -*-
"""
正向渲染引擎（Forward Renderer）：将逻辑自洽的 PlotNode 链表按顺序渲染成书。
换皮降维打击：严格套用 StyleGuide 文笔，将细纲扩写为正文并持久化到文件。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

try:
    from backend.engine.auto_novel_engine import assemble_rag_from_backend
    from backend.models.plot_graph import PlotGraphManager
    from backend.schemas.orchestrator_models import StyleGuide
    from backend.schemas.writing_schemas import WriteRequest, WriteResponse
    from backend.services.writing_service import WritingAgent
except ImportError:
    from engine.auto_novel_engine import assemble_rag_from_backend
    from models.plot_graph import PlotGraphManager
    from schemas.orchestrator_models import StyleGuide
    from schemas.writing_schemas import WriteRequest, WriteResponse
    from services.writing_service import WritingAgent

logger = logging.getLogger(__name__)

# 渲染时要求单章最少字数
RENDER_MIN_WORDS = 2000


class ForwardRenderer:
    """
    正向渲染流水线：获取全书 StyleGuide，按 previous_node_id / next_node_id 顺序遍历 PlotNode，
    对每个节点调用 WritingAgent 将细纲扩写为正文，并追加写入 output_file。
    """

    def __init__(
        self,
        *,
        writing_agent: WritingAgent,
        book_id: str = "",
    ) -> None:
        self.writing_agent = writing_agent
        self.book_id = book_id or ""

    def _build_render_instruction(self, style_guide: StyleGuide, node_summary: str, chapter_index: int) -> str:
        """组装单章渲染指令：文风 + 细纲，要求不少于 RENDER_MIN_WORDS 字、禁止发散。"""
        parts = [
            "【文风要求】",
            f"参考书名/风格: {style_guide.reference_book_name}",
        ]
        if style_guide.vocabulary_features:
            parts.append("可参考词汇/句式: " + "；".join(style_guide.vocabulary_features[:15]))
        if style_guide.pacing_rules:
            parts.append("行文节奏: " + style_guide.pacing_rules)
        if style_guide.dialogue_style:
            parts.append("对话风格: " + style_guide.dialogue_style)
        parts.append("")
        parts.append("【本章细纲】")
        parts.append(node_summary or "（无细纲，请按前文逻辑自然续写）")
        parts.append("")
        parts.append(
            f"请扮演一个文字渲染器：禁止发散剧情，严格使用上述文笔纹理，将本章细纲扩写为不少于 {RENDER_MIN_WORDS} 字的正文。"
        )
        return "\n".join(parts)

    async def render_full_book(
        self,
        book_id: str,
        output_file: str,
        *,
        style_guide: Optional[StyleGuide] = None,
    ) -> None:
        """
        渲染主循环：获取该书全局 StyleGuide，按链表顺序遍历 PlotNode，
        对每个节点调用 WritingAgent.generate_chapter，将正文追加写入 output_file。
        """
        self.book_id = book_id
        try:
            from backend.database import AsyncSessionLocal, init_sqlite_async
        except ImportError:
            from database import AsyncSessionLocal, init_sqlite_async
        await init_sqlite_async()

        if style_guide is None:
            try:
                from backend.engine.imitation_pipeline import load_style_guide_from_fingerprint_file
                from pathlib import Path
                root = Path(__file__).resolve().parents[2]
                fp_path = root / "data" / "cards" / book_id / "style_fingerprint.json"
                style_guide = load_style_guide_from_fingerprint_file(fp_path)
            except Exception as e:
                logger.warning("加载 StyleGuide 失败，使用空风格: %s", e)
                style_guide = StyleGuide(reference_book_name=book_id, vocabulary_features=[], pacing_rules="", dialogue_style="")
        if not style_guide:
            style_guide = StyleGuide(reference_book_name=book_id, vocabulary_features=[], pacing_rules="", dialogue_style="")

        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 清空或创建文件（若需追加可改为 "a"）
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("")

        async with AsyncSessionLocal() as session:
            manager = PlotGraphManager(session)
            nodes = await manager.iter_nodes_in_order(book_id)
        if not nodes:
            logger.warning("该书无 PlotNode 节点: book_id=%s", book_id)
            return

        for node in nodes:
            ch = node.chapter_index or 0
            summary = (node.summary or "").strip() or "（无细纲）"
            user_instruction = self._build_render_instruction(style_guide, summary, ch)
            request = WriteRequest(
                user_instruction=user_instruction,
                focus_character_ids=node.involved_characters or [],
                location_id=None,
            )
            response: WriteResponse = await self.writing_agent.generate_chapter(request, book_id=book_id)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n## 第{ch}章\n\n")
                f.write(response.draft_content)
            logger.info("第 %s 章渲染完成，文笔风格套用成功。", ch)
