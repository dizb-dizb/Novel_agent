# -*- coding: utf-8 -*-
"""
润色层 (Polish Layer)：指纹对齐，消除 AI 味。
对草稿做风格过滤：常用词汇频率、对话占比等，确保符合原书「文笔指纹」。
"""
import re
from typing import Any, Dict, Optional, Tuple

try:
    from src.librarian.style_store import StyleFingerprint
except ImportError:
    StyleFingerprint = None  # type: ignore


def _draft_stats(text: str) -> Dict[str, float]:
    """从草稿计算简易统计：段落均长、对话占比。"""
    if not text or not text.strip():
        return {"avg_paragraph_length": 0.0, "dialogue_ratio": 0.0, "char_count": 0}
    paras = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    total_chars = len(text)
    dialogue_chars = 0
    for p in paras:
        if re.search(r'[「『].*[」』]', p) or "说道" in p or "道：" in p:
            dialogue_chars += len(p)
    return {
        "avg_paragraph_length": sum(len(p) for p in paras) / len(paras) if paras else 0,
        "dialogue_ratio": dialogue_chars / total_chars if total_chars else 0,
        "char_count": total_chars,
    }


def style_fingerprint_check(
    draft: str,
    fingerprint: Optional["StyleFingerprint"] = None,
    tolerance: float = 0.25,
) -> Tuple[bool, str]:
    """
    对比草稿与全书风格指纹，判断是否通过并返回反馈。
    :param draft: 草稿正文
    :param fingerprint: 原书 StyleFingerprint；若为 None 则只做基础长度检查
    :param tolerance: 允许偏差比例，如 0.25 表示 ±25% 内视为通过
    :return: (passed, feedback)
    """
    stats = _draft_stats(draft)
    feedback_parts: list[str] = []
    passed = True

    if fingerprint:
        # 段落均长：允许在 (1 - tolerance) ~ (1 + tolerance) 倍范围内
        ref_para = fingerprint.avg_paragraph_length or 100
        curr_para = stats["avg_paragraph_length"]
        if ref_para > 0:
            ratio = curr_para / ref_para
            if ratio < 1 - tolerance or ratio > 1 + tolerance:
                passed = False
                feedback_parts.append(
                    f"段落均长偏离原书：当前约{curr_para:.0f}字/段，原书约{ref_para:.0f}字/段，建议调整断句节奏。"
                )
        # 对话占比
        ref_dial = fingerprint.dialogue_ratio
        curr_dial = stats["dialogue_ratio"]
        if ref_dial is not None and ref_dial >= 0:
            diff = abs(curr_dial - ref_dial)
            if diff > tolerance:
                passed = False
                feedback_parts.append(
                    f"对话占比偏离：当前约{curr_dial:.1%}，原书约{ref_dial:.1%}，建议增加或减少对话以贴近原书。"
                )
    else:
        # 无指纹时仅做基础检查：避免单段过长（AI 常见问题）
        if stats["avg_paragraph_length"] > 350:
            feedback_parts.append("存在过长段落，建议适当拆分以增强节奏感。")
            passed = False

    if not feedback_parts:
        feedback_parts.append("风格指纹检查通过，与原书节奏与对话比例基本一致。")
    return passed, "; ".join(feedback_parts)
