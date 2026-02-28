# -*- coding: utf-8 -*-
"""
草稿层 (Draft Layer)：风格化生成。
采用 Few-shot RAG：从 Librarian 调取原书最相似风格片段，引导模型生成 2000-3000 字正文。
国产模型选型：DeepSeek-Chat / Qwen-Plus（语感与性价比）。
"""
from typing import Any, Dict, List, Optional

from src.utils.llm_client import chat_low_cost

from .prompt_templates import get_template_for_chapter_type
from .state_schema import WriterState
from .style_injector import StyleInjector


def _target_length_from_reference(reference_chapter_length: Optional[int]) -> tuple[int, int]:
    """根据参考章节字数计算目标区间，使续写字数与原文相当。"""
    if not reference_chapter_length or reference_chapter_length <= 0:
        return 2000, 3500
    lo = max(1500, int(reference_chapter_length * 0.85))
    hi = min(6000, int(reference_chapter_length * 1.15))
    if lo >= hi:
        hi = lo + 500
    return lo, hi


def generate_draft_with_style(
    state: WriterState,
    style_injector: Optional[StyleInjector] = None,
    target_min_chars: int = 2000,
    target_max_chars: int = 3500,
    reference_chapter_length: Optional[int] = None,
) -> tuple[str, List[Dict[str, Any]]]:
    """
    根据节拍表与约束边界，结合风格样本生成正文草稿。
    若提供 reference_chapter_length，则目标字数与之相当（约 ±15%），便于读者体验一致。
    :return: (draft_text, style_samples_used)
    """
    if reference_chapter_length is not None:
        target_min_chars, target_max_chars = _target_length_from_reference(reference_chapter_length)
    injector = style_injector or StyleInjector(book_id=state.book_id)
    style_block = injector.get_style_prompt_block(
        chapter_type=state.chapter_type,
        tags=None,
        max_samples=5,
        max_chars_per_sample=400,
    )
    fingerprint_block = ""
    if injector.style_store:
        fp = injector.style_store.get_fingerprint()
        if fp:
            parts = []
            if fp.writing_habits:
                parts.append(f"写作习惯：{fp.writing_habits}")
            if fp.sentence_style:
                parts.append(f"句式特点：{fp.sentence_style}")
            if fp.rhetoric_notes:
                parts.append(f"修辞与语气：{fp.rhetoric_notes}")
            desc = getattr(fp, "representative_descriptions", None) or []
            if desc:
                parts.append("## 原文代表性描写片段（请模仿其笔触与节奏）")
                for i, d in enumerate(desc[:5], 1):
                    parts.append(f"【描写{i}】\n{d[:400]}{'…' if len(d) > 400 else ''}")
            speech = getattr(fp, "character_speech_samples", None) or []
            if speech:
                parts.append("## 角色代表性说话习惯（写对话时请贴合其口吻）")
                for s in speech[:4]:
                    role = (s.get("role") if isinstance(s, dict) else "") or "角色"
                    sample = (s.get("sample") if isinstance(s, dict) else "") or ""
                    if sample:
                        parts.append(f"【{role}】{sample[:300]}{'…' if len(sample) > 300 else ''}")
            if parts:
                fingerprint_block = "\n## 写作风格指纹（请拟合上述手法与习惯）\n" + "\n\n".join(parts) + "\n"
    template_extra = get_template_for_chapter_type(state.chapter_type)
    user = f"""请根据以下「节拍表」与「约束边界」写一章正文，字数在 {target_min_chars}-{target_max_chars} 字之间。

节拍表：
{state.beat_sheet}

约束边界：
{state.constraint_boundaries or "无额外约束"}
"""
    if state.preference_patch:
        user += f"\n## 用户偏好（写作时务必体现）\n{state.preference_patch}\n"
    if state.steering_hint:
        user += f"\n当前用户期待走向：{state.steering_hint}\n"
    if fingerprint_block:
        user += fingerprint_block
    user += f"""
{template_extra}

{style_block}

请直接输出正文，不要输出「节拍表」或「第一章」等标题，从第一个场景开始写。"""
    messages = [
        {"role": "system", "content": "你是网文写手，严格按节拍表与风格样本写作，只输出正文。"},
        {"role": "user", "content": user},
    ]
    draft = chat_low_cost(messages)
    draft = (draft or "").strip()
    samples_used = [
        {"chapter_index": getattr(s, "chapter_index", 0), "preview": (getattr(s, "text", "") or "")[:80]}
        for s in (injector.style_store.samples[:5] if injector.style_store else [])
    ]
    return draft, samples_used
