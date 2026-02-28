# -*- coding: utf-8 -*-
"""
风格迁移组件：从向量库/StyleStore 动态召回「风格 DNA」并注入提示词。
供草稿层 Few-shot RAG 使用，确保措辞、语气、断句与原作者对齐。
"""
from typing import Any, Dict, List, Optional

try:
    from src.librarian.style_store import StyleStore, StyleSample
except ImportError:
    StyleStore = None  # type: ignore
    StyleSample = None  # type: ignore


class StyleInjector:
    """
    从 Librarian 召回与当前场景最相似的风格片段，并格式化为 Few-shot 提示块。
    """

    def __init__(self, style_store: Optional["StyleStore"] = None, book_id: str = ""):
        if style_store is None and StyleStore is not None:
            style_store = StyleStore(book_id=book_id)
        self.style_store = style_store

    def get_style_prompt_block(
        self,
        chapter_type: str = "",
        tags: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        max_samples: int = 5,
        max_chars_per_sample: int = 400,
    ) -> str:
        """
        根据章节类型/标签/关键词召回样本，拼成「风格样本」提示块。
        """
        if not self.style_store or not self.style_store.samples:
            return "（暂无风格样本，请按常规网文风格写作）"
        if tags is None:
            tags = _chapter_type_to_tags(chapter_type)
        samples = self.style_store.get_samples_by_tags(tags, limit=max_samples)
        if not samples and keywords:
            samples = self.style_store.get_samples_by_keywords(keywords, limit=max_samples)
        if not samples:
            samples = self.style_store.samples[:max_samples]
        block = "## 原书风格样本（请模仿其措辞、节奏与断句）\n\n"
        for i, s in enumerate(samples[:max_samples], 1):
            text = (s.text or "")[:max_chars_per_sample]
            if len(s.text or "") > max_chars_per_sample:
                text += "…"
            block += f"【样本{i}】\n{text}\n\n"
        return block.strip()

    def inject_into_messages(
        self,
        system_base: str,
        chapter_type: str = "",
        tags: Optional[List[str]] = None,
    ) -> str:
        """将风格样本块追加到系统提示末尾。"""
        block = self.get_style_prompt_block(chapter_type=chapter_type, tags=tags)
        return system_base.rstrip() + "\n\n" + block


def _chapter_type_to_tags(chapter_type: str) -> List[str]:
    m = {
        "战斗章": ["#打斗"],
        "感情章": ["#心理", "#表白", "#对话"],
        "日常章": ["#日常", "#对话"],
        "悬念章": ["#悬念", "#环境"],
    }
    return m.get(chapter_type, ["#对话", "#环境"])
