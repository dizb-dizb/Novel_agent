# -*- coding: utf-8 -*-
"""
裁判 (Simulator AI)：自动化评分，盲测对比原著与 AI 续写。
"""
import json
import random
from typing import Optional

from pydantic import BaseModel, Field

from src.utils.llm_client import chat_high_quality


class EvaluationResult(BaseModel):
    """评估结果：三维度分数 + 评语。"""
    style_score: int = Field(default=0, ge=0, le=100, description="文风拟合度")
    logic_score: int = Field(default=0, ge=0, le=100, description="逻辑连贯性")
    coherence_score: int = Field(default=0, ge=0, le=100, description="剧情推动力")
    critique: str = Field(default="", description="具体评语")
    which_better: Optional[str] = Field(default=None, description="盲测时：A 或 B 哪个更好")
    is_original_first: Optional[bool] = Field(default=None, description="盲测时：原著是否在 A")


class SimulatorAI:
    """
    模拟读者：对比原著与 AI 续写，从文风、逻辑、剧情推动力三维度评分。
    盲测：随机打乱 A/B 顺序，减少位置偏见。
    """

    def __init__(self, use_blind: bool = True):
        self.use_blind = use_blind

    def evaluate(
        self,
        original_text: str,
        generated_text: str,
        max_chars_each: int = 2500,
    ) -> EvaluationResult:
        """
        输入两段文本，返回 EvaluationResult。
        若 use_blind 为 True，则随机将两段标为 A/B 再送评，最后把分数映射回「原著 / 续写」。
        """
        orig = (original_text or "")[:max_chars_each]
        gen = (generated_text or "")[:max_chars_each]
        if not orig and not gen:
            return EvaluationResult(critique="无文本可评")

        # 盲测：随机决定 A/B 谁对应原著
        original_first = random.choice([True, False])
        if original_first:
            text_a, text_b = orig, gen
        else:
            text_a, text_b = gen, orig

        user = f"""你是一位极其挑剔的资深网文编辑。请对比下面两段续写文本（A 和 B），其中一段是原著续写，一段是 AI 生成。请从三个维度分别对 **A** 和 **B** 打分（0-100），并说明哪段更好及理由。

【文本 A】
{text_a}

【文本 B】
{text_b}

请严格按以下 JSON 格式输出（仅此 JSON，不要 markdown 包裹）：
{{
  "A": {{ "style_score": 0-100, "logic_score": 0-100, "coherence_score": 0-100 }},
  "B": {{ "style_score": 0-100, "logic_score": 0-100, "coherence_score": 0-100 }},
  "which_better": "A 或 B",
  "critique": "一段简短评语：文风、逻辑、剧情推动力各有什么问题或亮点"
}}

说明：
- style_score：文风拟合度（用词、句式长度、语气是否像网文）
- logic_score：逻辑连贯性（是否前后矛盾、人设是否崩）
- coherence_score：剧情推动力（是否水字数、是否有推进）
"""
        messages = [
            {"role": "system", "content": "你是资深网文编辑，只输出上述 JSON。"},
            {"role": "user", "content": user},
        ]
        raw = chat_high_quality(messages)
        raw = (raw or "").strip()
        for p in ("```json", "```"):
            if raw.startswith(p):
                raw = raw[len(p):].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        try:
            data = json.loads(raw)
        except Exception:
            return EvaluationResult(critique="解析评分失败")

        # 我们要的是「续写（generated）」的分数；若原著在 A，则 B 是续写
        a_scores = data.get("A") or {}
        b_scores = data.get("B") or {}
        which = (data.get("which_better") or "").upper()
        if "A" in which:
            which_better = "A"
        elif "B" in which:
            which_better = "B"
        else:
            which_better = "A"

        # 续写段落在 A 还是 B？
        gen_is_a = not original_first
        if gen_is_a:
            style = int(a_scores.get("style_score", 50))
            logic = int(a_scores.get("logic_score", 50))
            coherence = int(a_scores.get("coherence_score", 50))
        else:
            style = int(b_scores.get("style_score", 50))
            logic = int(b_scores.get("logic_score", 50))
            coherence = int(b_scores.get("coherence_score", 50))

        return EvaluationResult(
            style_score=min(100, max(0, style)),
            logic_score=min(100, max(0, logic)),
            coherence_score=min(100, max(0, coherence)),
            critique=data.get("critique") or "",
            which_better=which_better,
            is_original_first=original_first,
        )

    @staticmethod
    def average_score(result: EvaluationResult) -> float:
        """三维度平均分。"""
        return (result.style_score + result.logic_score + result.coherence_score) / 3.0
