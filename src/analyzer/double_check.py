# -*- coding: utf-8 -*-
"""
逻辑与风格的双重校对门：续写稿需通过逻辑审核 + 风格审核后方可采纳。
"""
import re
from typing import Any, Dict, List, Optional

from src.utils.llm_client import chat_high_quality

from .state_schema import AnalysisState

try:
    from src.librarian.style_store import StyleSample
except ImportError:
    from librarian.style_store import StyleSample  # type: ignore


def logic_check(
    state: AnalysisState,
    draft_text: str,
    chapter_index: int,
    context_summary: str = "",
) -> Dict[str, Any]:
    """
    逻辑审核：高质量模型检查续写内容是否产生逻辑漏洞（如瞬移、违背前文设定等）。
    :return: { "passed": bool, "conflicts": [str], "suggestion": str }
    """
    from .backtrack import _state_context
    ctx = _state_context(state, max_cards=60, max_nodes=80)
    user = f"""以下是一段续写正文（发生在第 {chapter_index + 1} 章之后），请检查是否与已有设定/因果矛盾。

当前世界观与因果摘要：
{ctx}

续写正文（前 2000 字）：
{draft_text[:2000]}

请输出一个 JSON 对象，且仅此 JSON：
{{ "passed": true/false, "conflicts": ["矛盾1", "矛盾2"], "suggestion": "若未通过，给出修改建议" }}
"""
    messages = [
        {"role": "system", "content": "你是小说逻辑审核员，只输出 JSON。"},
        {"role": "user", "content": user},
    ]
    raw = chat_high_quality(messages)
    raw = raw.strip()
    for p in ("```json", "```"):
        if raw.startswith(p):
            raw = raw[len(p):].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    try:
        import json
        data = json.loads(raw)
        return {
            "passed": data.get("passed", False),
            "conflicts": data.get("conflicts") or [],
            "suggestion": data.get("suggestion") or "",
        }
    except Exception:
        return {"passed": False, "conflicts": ["无法解析审核结果"], "suggestion": ""}


def _simple_style_score(draft: str, samples: List[StyleSample]) -> float:
    """
    简易风格相似度：基于标点与句长的粗糙一致性（无向量时使用）。
    返回 0~1，越高越接近样本风格。
    """
    if not samples or not draft.strip():
        return 1.0
    draft_sents = [s for s in re.split(r"[。！？\n]+", draft) if len(s.strip()) > 2]
    sample_sents = []
    for s in samples[:5]:
        sample_sents.extend([x for x in re.split(r"[。！？\n]+", s.text) if len(x.strip()) > 2])
    if not draft_sents or not sample_sents:
        return 0.8
    avg_len_draft = sum(len(s) for s in draft_sents) / len(draft_sents)
    avg_len_sample = sum(len(s) for s in sample_sents) / len(sample_sents)
    len_ratio = min(avg_len_draft, avg_len_sample) / max(avg_len_draft, avg_len_sample) if max(avg_len_draft, avg_len_sample) > 0 else 1.0
    return round(min(1.0, len_ratio + 0.3), 2)


def style_check(
    draft_text: str,
    style_samples: List[StyleSample],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    风格审核：比较续写与原书样本的相似度，若偏差过大则打回。
    :param threshold: 简易得分低于此则未通过
    :return: { "passed": bool, "score": float, "suggestion": str }
    """
    score = _simple_style_score(draft_text, style_samples)
    passed = score >= threshold
    suggestion = "" if passed else "建议增加与原书相近的句式长度与节奏，或补充风格样本后再审。"
    return {
        "passed": passed,
        "score": score,
        "suggestion": suggestion,
    }


def double_check_gate(
    state: AnalysisState,
    draft_text: str,
    chapter_index: int,
    style_samples: Optional[List[StyleSample]] = None,
    style_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    双重校对：先逻辑审核，再风格审核；任一未通过则返回整体未通过。
    """
    logic_result = logic_check(state, draft_text, chapter_index)
    style_result = style_check(draft_text, style_samples or [], threshold=style_threshold)
    passed = logic_result["passed"] and style_result["passed"]
    return {
        "passed": passed,
        "logic": logic_result,
        "style": style_result,
        "suggestion": "; ".join(
            filter(None, [logic_result.get("suggestion"), style_result.get("suggestion")])
        ),
    }
