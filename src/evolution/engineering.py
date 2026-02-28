# -*- coding: utf-8 -*-
"""
医生 (Engineering Diagnostician)：根据日志与评分反推代码问题，产出 Markdown Action Plan。
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.llm_client import chat_high_quality

from .simulator import EvaluationResult
from .trace_logger import TraceEvent, TraceLogger


def _read_code_context(writer_dir: Path, librarian_dir: Path, max_lines: int = 800) -> str:
    """读取 writer 与 librarian 相关代码片段。"""
    parts = []
    for base in [writer_dir, librarian_dir]:
        if not base.is_dir():
            continue
        for f in sorted(base.glob("*.py"))[:8]:
            try:
                text = f.read_text(encoding="utf-8")
                parts.append(f"## {f.name}\n```\n{text[:4000]}\n```")
            except Exception:
                pass
    combined = "\n\n".join(parts)
    if len(combined) > max_lines * 80:
        combined = combined[: max_lines * 80] + "\n...(truncated)"
    return combined


class EngineeringDiagnostician:
    """
    工程诊断专家：结合低分评价、完整 trace 日志与代码上下文，输出 Markdown Action Plan。
    CoT：检索检查 → 指令检查 → 模型检查。
    """

    def __init__(
        self,
        writer_dir: Optional[Path] = None,
        librarian_dir: Optional[Path] = None,
    ):
        root = Path(__file__).resolve().parents[1]
        self.writer_dir = writer_dir or root / "writer"
        self.librarian_dir = librarian_dir or root / "librarian"

    def diagnose(
        self,
        evaluation: EvaluationResult,
        trace_logs: List[TraceEvent],
        code_context: Optional[str] = None,
    ) -> str:
        """
        输入：Simulator 低分评价、该次生成的完整日志、可选代码上下文。
        输出：Markdown 格式的 Action Plan（Problem Analysis + Code Modification Suggestion）。
        """
        code = code_context or _read_code_context(self.writer_dir, self.librarian_dir)
        eval_summary = (
            f"- style_score: {evaluation.style_score}\n"
            f"- logic_score: {evaluation.logic_score}\n"
            f"- coherence_score: {evaluation.coherence_score}\n"
            f"- critique: {evaluation.critique}"
        )
        trace_summary = []
        for i, ev in enumerate(trace_logs[-15:]):
            trace_summary.append(
                f"[{i+1}] module={ev.module}\n"
                f"  inputs keys: {list((ev.inputs or {}).keys())}\n"
                f"  context_snapshot length: {len(ev.context_snapshot or [])}\n"
                f"  error: {ev.error or 'none'}"
            )
        trace_block = "\n".join(trace_summary)

        user = f"""你是一个工程诊断专家。根据以下「低分评估结果」「生成过程日志」和「相关代码上下文」，分析问题根源并给出可执行的代码修改建议。

## 评估结果
{eval_summary}

## 生成过程日志（最近若干条）
{trace_block}

## 相关代码（writer / librarian 片段）
{code[:12000]}

请按以下结构输出一份 **Markdown 格式的 Action Plan**（不要输出其他无关内容）：

---
## Problem Analysis
（简要写出：问题根源是检索不足 / 指令不明确 / 模型或参数不合适 等，并引用日志中的证据）

## Code Modification Suggestion
（具体修改建议，精确到**文件名**和**函数名**。例如：在 `src/writer/prompt_templates.py` 的 `get_template_for_chapter_type` 中增加风格权重的说明；或建议将 `src/librarian/context_loader.py` 中的 top_k 从 5 调整为 10）
---

只输出上述 Markdown，不要用 ```markdown 包裹。
"""
        messages = [
            {"role": "system", "content": "你是代码与生成流程诊断专家，只输出 Markdown Action Plan。"},
            {"role": "user", "content": user},
        ]
        raw = chat_high_quality(messages)
        return (raw or "").strip()
