# -*- coding: utf-8 -*-
"""
意图解析器（Intent Analyzer）：实时解析用户当前的续写/改写意图。
将自然语言转为结构化 ParsedIntent，供 Planner 与 User Expert 使用。
"""
import json
import re
from typing import Any, Dict, List, Optional

from src.utils.llm_client import chat_high_quality

from .schema import ParsedIntent


# 意图类型
INTENT_CONTINUE = "continue"      # 续写下一章/按原逻辑走
INTENT_REWRITE = "rewrite"       # 改写某章/某段
INTENT_SIMILAR = "similar_book"   # 想要类似体验/换皮新书
INTENT_ASK = "ask"               # 追问/选择走向
INTENT_OTHER = "other"


def parse_intent(
    raw_text: str,
    book_context: str = "",
    allow_llm: bool = True,
) -> ParsedIntent:
    """
    解析用户输入为结构化意图。
    :param raw_text: 用户原话
    :param book_context: 可选，当前书籍/章节简要上下文
    :param allow_llm: 是否用 LLM 做深度解析，否则仅规则
    """
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return ParsedIntent(intent_type=INTENT_OTHER, raw_text="", summary="", confidence=0.0)

    # 规则快速判断
    lower = raw_text.lower()
    if any(k in raw_text for k in ["续写", "接着写", "下一章", "后面怎样"]):
        base = ParsedIntent(
            intent_type=INTENT_CONTINUE,
            raw_text=raw_text,
            summary=raw_text[:100],
            slots={"continue": True},
            confidence=0.85,
        )
        if allow_llm and len(raw_text) > 20:
            return _refine_intent_with_llm(raw_text, base, book_context)
        return base
    if any(k in raw_text for k in ["改写", "重写", "改成", "把第"]):
        base = ParsedIntent(
            intent_type=INTENT_REWRITE,
            raw_text=raw_text,
            summary=raw_text[:100],
            slots={"rewrite": True},
            confidence=0.8,
        )
        if allow_llm and len(raw_text) > 15:
            return _refine_intent_with_llm(raw_text, base, book_context)
        return base
    if any(k in raw_text for k in ["类似", "同类型", "换一本", "还想看这种", "同款"]):
        base = ParsedIntent(
            intent_type=INTENT_SIMILAR,
            raw_text=raw_text,
            summary=raw_text[:100],
            slots={"similar": True},
            confidence=0.75,
        )
        if allow_llm:
            return _refine_intent_with_llm(raw_text, base, book_context)
        return base
    if any(k in raw_text for k in ["希望", "想要", "能不能", "是否", "还是"]):
        base = ParsedIntent(
            intent_type=INTENT_ASK,
            raw_text=raw_text,
            summary=raw_text[:100],
            slots={"ask": True},
            confidence=0.7,
        )
        if allow_llm:
            return _refine_intent_with_llm(raw_text, base, book_context)
        return base

    if allow_llm and len(raw_text) >= 10:
        return _refine_intent_with_llm(
            raw_text,
            ParsedIntent(intent_type=INTENT_OTHER, raw_text=raw_text, summary=raw_text[:80], confidence=0.5),
            book_context,
        )
    return ParsedIntent(
        intent_type=INTENT_CONTINUE,
        raw_text=raw_text,
        summary=raw_text[:100],
        slots={"freeform": raw_text},
        confidence=0.6,
    )


def _refine_intent_with_llm(raw_text: str, base: ParsedIntent, book_context: str) -> ParsedIntent:
    """用 LLM 细化 intent_type、summary、slots。"""
    user = f"""用户说：「{raw_text}」
{f'当前书籍/章节上下文：{book_context[:300]}' if book_context else ''}

请判断用户意图类型并输出一个 JSON（仅此 JSON）：
- intent_type: "continue" | "rewrite" | "similar_book" | "ask" | "other"
- summary: 一句话概括用户想做什么（20字内）
- slots: 对象，可包含 target_chapter（若提到章节）、focus_character、rewrite_scope、similar_genre 等
- confidence: 0~1
"""
    msgs = [
        {"role": "system", "content": "你只输出一个 JSON 对象，不要 markdown。"},
        {"role": "user", "content": user},
    ]
    raw = (chat_high_quality(msgs) or "").strip()
    for p in ("```json", "```"):
        if raw.startswith(p):
            raw = raw[len(p):].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    try:
        data = json.loads(raw)
    except Exception:
        return base
    return ParsedIntent(
        intent_type=data.get("intent_type") or base.intent_type,
        raw_text=raw_text,
        summary=data.get("summary") or base.summary,
        slots=data.get("slots") or base.slots,
        confidence=float(data.get("confidence", base.confidence)),
    )


def extract_continuation_intent_for_writer(parsed: ParsedIntent) -> str:
    """
    从 ParsedIntent 抽出一句可直接作为 WriterState.user_intent 的续写意图。
    续写场景下优先用 summary；若为 ask 则转为「用户希望…」。
    """
    if parsed.intent_type == INTENT_ASK and parsed.raw_text:
        return f"用户希望：{parsed.summary or parsed.raw_text[:150]}"
    if parsed.slots.get("freeform"):
        return str(parsed.slots["freeform"])[:300]
    return parsed.summary or parsed.raw_text[:200]
