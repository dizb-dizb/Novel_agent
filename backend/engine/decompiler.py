# -*- coding: utf-8 -*-
"""
反编译器（Book Decompiler）：逆向拆解原著，将 txt/章节列表 吐成结构化知识库 + 情节链表。
吞噬引擎：吃掉全书正文，产出四大知识库（角色、设定、关系、情节）与 PlotNode 链表。
"""
from __future__ import annotations

import json
import logging
import random
from typing import Any, Dict, List, Optional

try:
    from backend.engine.auto_novel_engine import AnalysisAgent
    from backend.engine.logic_master import LogicMasterAgent
    from backend.models.plot_graph import PlotNode
    from backend.schemas.orchestrator_models import StyleGuide
except ImportError:
    from engine.auto_novel_engine import AnalysisAgent
    from engine.logic_master import LogicMasterAgent
    from models.plot_graph import PlotNode
    from schemas.orchestrator_models import StyleGuide

logger = logging.getLogger(__name__)

# 单章提取重试
DECOMPILE_RETRY = 3
# 文风采样章节数
STYLE_SAMPLE_SIZE = 5


class BookDecompiler:
    """
    逆向解析已有原著：文风提取 + 全本逐章扫描，产出 StyleGuide 与情节链表（PlotNode）+ 四大知识库更新。
    """

    def __init__(
        self,
        *,
        analysis_agent: AnalysisAgent,
        logic_master: LogicMasterAgent,
        book_id: str = "",
    ) -> None:
        self.analysis_agent = analysis_agent
        self.logic_master = logic_master
        self.book_id = book_id or ""

    async def extract_global_style(self, sample_chapters: List[str]) -> StyleGuide:
        """
        从原著中随机抽取 3～5 章作为样本，调用大模型提取高频词汇、句式结构、对话风格，存入全局 StyleGuide。
        """
        if not sample_chapters:
            return StyleGuide(
                reference_book_name=self.book_id or "未知",
                vocabulary_features=[],
                pacing_rules="",
                dialogue_style="",
            )
        k = min(STYLE_SAMPLE_SIZE, len(sample_chapters))
        indices = random.sample(range(len(sample_chapters)), k)
        combined = "\n\n---\n\n".join(
            f"## 第{i+1}章\n{(sample_chapters[i] or '')[:4000]}"
            for i in sorted(indices)
        )
        return await self.logic_master.extract_style_guide_async(combined)

    async def _extract_chapter_node(
        self,
        book_id: str,
        chapter_index: int,
        chapter_text: str,
    ) -> Dict[str, Any]:
        """
        单章抽取：返回本章的 summary、involved_characters 等，供写入 PlotNode。
        若 LLM 返回非法 JSON 则抛出，由上层重试。
        """
        import os
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError("请安装 openai")
        try:
            from src.utils.llm_client import get_model_config
            base_url, api_key, model_id = get_model_config(
                os.getenv("ANALYZER_LOW_MODEL", "glm-4.5-air")
            )
        except ImportError:
            base_url, api_key, model_id = "https://open.bigmodel.cn/api/paas/v4", os.getenv("ZHIPU_API_KEY") or os.getenv("OPENAI_API_KEY"), "glm-4.5-air"
        if not api_key:
            return {"summary": (chapter_text or "")[:500].strip() or f"第{chapter_index}章", "involved_characters": None}
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        prompt = f"""请对以下网文章节做极简抽取，只输出一个 JSON 对象（不要 markdown 包裹）：
{{
  "summary": "本章情节概要（80～200 字）",
  "involved_characters": ["角色名1", "角色名2"]
}}

章节正文：
{(chapter_text or "")[:6000]}
"""
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (resp.choices[0].message.content or "").strip()
        for prefix in ("```json", "```"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        data = json.loads(raw)
        return {
            "summary": (data.get("summary") or "")[:2000] or (chapter_text or "")[:500],
            "involved_characters": data.get("involved_characters") if isinstance(data.get("involved_characters"), list) else None,
        }

    async def decompile_book(
        self,
        book_id: str,
        book_text_by_chapters: List[str],
        *,
        style_guide_out: Optional[StyleGuide] = None,
    ) -> StyleGuide:
        """
        全本扫描流水线：遍历每一章，调用 AnalysisAgent 提取并更新角色/设定/关系/情节，
        同时构建 PlotNode 链表（previous_node_id / next_node_id）。某章抽取失败时重试，不中断全书。
        """
        self.book_id = book_id
        try:
            from backend.database import AsyncSessionLocal, init_sqlite_async
        except ImportError:
            from database import AsyncSessionLocal, init_sqlite_async
        await init_sqlite_async()

        # 文风：用 3～5 章样本
        sample_size = min(STYLE_SAMPLE_SIZE, len(book_text_by_chapters))
        sample_chapters = book_text_by_chapters[:sample_size] if book_text_by_chapters else []
        style_guide = style_guide_out or await self.extract_global_style(sample_chapters)
        logger.info("全局文风已提取: reference_book_name=%s", style_guide.reference_book_name)

        prev_id: Optional[str] = None
        async with AsyncSessionLocal() as session:
            for i, text in enumerate(book_text_by_chapters):
                ch_num = i + 1
                node_data: Optional[Dict[str, Any]] = None
                for attempt in range(DECOMPILE_RETRY):
                    try:
                        node_data = await self._extract_chapter_node(book_id, ch_num, text)
                        break
                    except (json.JSONDecodeError, KeyError, Exception) as e:
                        logger.warning("反编译第 %s 章抽取失败（第 %s 次）: %s", ch_num, attempt + 1, e)
                        if attempt == DECOMPILE_RETRY - 1:
                            node_data = {
                                "summary": (text or "")[:500].strip() or f"第{ch_num}章",
                                "involved_characters": None,
                            }
                if not node_data:
                    node_data = {"summary": f"第{ch_num}章", "involved_characters": None}

                node = PlotNode(
                    book_id=book_id,
                    chapter_index=ch_num,
                    summary=node_data.get("summary"),
                    involved_characters=node_data.get("involved_characters"),
                    previous_node_id=prev_id,
                    next_node_id=None,
                    sequence_order=i,
                    is_mutated=False,
                    needs_review=False,
                )
                session.add(node)
                await session.flush()
                if prev_id:
                    prev = await session.get(PlotNode, prev_id)
                    if prev:
                        prev.next_node_id = node.id
                prev_id = node.id

                await self.analysis_agent.extract_and_update_db(book_id, ch_num, text or "")
                logger.info("正在反编译第 %s 章... 节点已写入，知识库已更新。", ch_num)

            await session.commit()

        return style_guide
