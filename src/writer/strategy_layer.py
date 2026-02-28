# -*- coding: utf-8 -*-
"""
战略层 (Strategy Layer)：生成本章「剧情节拍表 (Beat Sheet)」，不写正文。
对比 Librarian 元协议，使用高质量模型（DeepSeek-V3 / Qwen-Max）进行逻辑编排。
"""
import json
from typing import Optional

from src.utils.llm_client import chat_high_quality

from .prompt_templates import get_template_for_chapter_type
from .state_schema import WriterState


def generate_beat_sheet(
    state: WriterState,
    max_beats: int = 8,
) -> tuple[str, str]:
    """
    接收改写/续写意图与三维上下文，生成本章的剧情节拍表及建议章节类型。
    :return: (beat_sheet 文本, chapter_type)
    """
    ctx = f"""
## 历史因果（有效伏笔与节点）
{state.history_causal_pack[:3000]}

## 新因果锚点（用户改写/续写起点）
{state.new_causal_anchors}

## 规则约束（世界观红线）
{state.rule_constraints_pack[:1500]}
"""
    preference_block = ""
    if state.preference_patch:
        preference_block = f"\n## 用户偏好（请在大纲中体现）\n{state.preference_patch}\n"
    steering_block = f"\n当前用户期待走向：{state.steering_hint}\n" if state.steering_hint else ""
    user = f"""你是一位网文大纲师。请根据以下信息，为「第 {state.chapter_index + 1} 章」生成本章的**剧情节拍表 (Beat Sheet)**，不要写正文。

用户意图：{state.user_intent}

{ctx}
{preference_block}{steering_block}
要求：
1. 输出 3-{max_beats} 个节拍，每节用一句话概括（例如：第1节：主角潜入；第2节：发现密信（伏笔A）；第3节：遭遇小怪）。
2. 节拍需符合上述因果与规则，不违背世界观红线。
3. 最后一行单独标明本章建议类型，从 [战斗章/感情章/日常章/悬念章] 中选一，格式：章节类型：xxx。

只输出节拍表与类型，不要其他解释。"""
    messages = [
        {"role": "system", "content": "你只输出剧情节拍表与一行章节类型，不写正文，不解释。"},
        {"role": "user", "content": user},
    ]
    raw = chat_high_quality(messages)
    raw = (raw or "").strip()
    chapter_type = ""
    for label in ["战斗章", "感情章", "日常章", "悬念章"]:
        if f"章节类型：{label}" in raw or f"章节类型:{label}" in raw:
            chapter_type = label
            raw = raw.replace(f"章节类型：{label}", "").replace(f"章节类型:{label}", "").strip()
            break
    return raw, chapter_type
