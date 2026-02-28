# -*- coding: utf-8 -*-
"""
WritingAgent：网文续写核心引擎。
RAG 组装（情节/设定/角色/关系）→ CoT + 正文生成 → 审查重试 → 偏好数据沉淀。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

try:
    from backend.schemas.writing_schemas import WriteRequest, WriteResponse
except ImportError:
    from schemas.writing_schemas import WriteRequest, WriteResponse


# ---------- 异常 ----------
class OutputFormatError(Exception):
    """LLM 未输出规定的 <thought_process> / <draft> 标签。"""
    pass


# ---------- 系统提示词 ----------
# 推理阶段（DeepSeek R1）：只要求输出思考过程
SYSTEM_PROMPT = """你是一位顶尖的网文写手，擅长感官推演、动机博弈与动作微操。

请根据给定的 RAG 上下文与用户指令，在 <thought_process> 标签内完成感官推演、动机博弈和动作微操的思考。
只输出 <thought_process>...</thought_process>，不要写正文。思考要具体到场景、人物动机与冲突设计。"""

# 文笔阶段（Qwen-max）：根据思考过程只输出正文，篇幅须符合用户指令
DRAFT_SYSTEM_PROMPT = """你是一位顶尖的网文写手。根据给定的思考过程与上下文，仅在 <draft> 标签内输出网文正文。
正文字数须严格符合用户指令中的篇幅要求（若指令中有「约 N 字」则本章正文应接近 N 字，不得明显偏短或偏长）；若未指定篇幅则不少于 1000 字。
不要输出 <thought_process> 或其它内容，只输出 <draft>...</draft>。"""


# ---------- WritingAgent ----------
class WritingAgent:
    """网文续写 Agent：RAG 上下文组装 → LLM 生成（CoT + 正文）→ 解析 → 审查重试 → 数据沉淀。"""

    def __init__(
        self,
        *,
        reasoning_model: Optional[str] = None,
        draft_model: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dataset_path: Optional[Path] = None,
        context_assembler: Optional[Callable[[WriteRequest, Optional[str]], Awaitable[str]]] = None,
    ) -> None:
        # 模型解耦：逻辑推理默认 deepseek-reasoner，文笔默认 qwen-max（可由环境变量覆盖）
        self.reasoning_model = (
            reasoning_model
            or os.getenv("WRITING_REASONING_MODEL")
            or os.getenv("ANALYZER_HIGH_MODEL")
            or "deepseek-reasoner"
        ).strip()
        self.draft_model = (
            draft_model or os.getenv("WRITING_DRAFT_MODEL") or "qwen-max"
        ).strip()
        # 兼容旧参数：若显式传入 model/api_key/base_url 则作为单模型回退
        self._legacy_model = (model or os.getenv("WRITING_AGENT_MODEL") or "").strip()
        self._legacy_api_key = (api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or "").strip()
        self._legacy_base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "").strip() or None
        self.dataset_path = Path(dataset_path or os.getenv("SFT_DATASET_PATH", "data/sft_training_dataset.jsonl"))
        self.context_assembler = context_assembler

    def _get_model_config(self, model_name: str) -> tuple[Optional[str], Optional[str], str]:
        """解析模型名 -> (base_url, api_key, model_id)。后端独立运行时回退到 env。"""
        try:
            from src.utils.llm_client import get_model_config
            base_url, key, model_id = get_model_config(model_name)
            return base_url, key or "", model_id
        except ImportError:
            pass
        # 后端无 src 时：仅支持推理/文笔两模型
        fallback = {
            "deepseek-reasoner": (
                "https://api.deepseek.com",
                os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
                "deepseek-reasoner",
            ),
            "qwen-max": (
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
                "qwen-max",
            ),
        }
        return fallback.get(model_name, (self._legacy_base_url, self._legacy_api_key, model_name))

    async def _assemble_context(self, request: WriteRequest, book_id: Optional[str] = None) -> str:
        """
        组装 RAG 上下文：若已注入 context_assembler 且提供 book_id，则从四大知识库拉取；
        否则使用内置占位（供单机测试）。
        """
        if self.context_assembler and book_id:
            return await self.context_assembler(request, book_id)
        lines = [
            "[环境法则]",
            "  - 世界观与当前场景规则（由设定库提供）",
            "[出场人物性格]",
            "  - 关注角色列表: " + (", ".join(request.focus_character_ids) if request.focus_character_ids else "无"),
            "  - 性格与说话习惯（由角色库提供）",
            "[人物历史恩怨]",
            "  - 关系与好感度（由关系图谱提供）",
            "[前情提要]",
            "  - 近期情节因果（由情节树提供）",
        ]
        if request.location_id:
            lines.insert(2, f"  - 发生地点ID: {request.location_id}")
        return "\n".join(lines)

    def _parse_llm_response(self, text: str) -> tuple[str, str]:
        """
        从 LLM 返回中提取 <thought_process> 和 <draft> 标签内容。
        :return: (thought_process, draft_content)
        :raises OutputFormatError: 缺少标签时
        """
        text = (text or "").strip()
        thought = ""
        draft = ""

        m = re.search(r"<thought_process\s*>(.*?)</thought_process\s*>", text, re.DOTALL | re.IGNORECASE)
        if m:
            thought = m.group(1).strip()
        m_thought = re.search(r"<thought\s*>(.*?)</thought\s*>", text, re.DOTALL | re.IGNORECASE)
        if m_thought and not thought:
            thought = m_thought.group(1).strip()

        m = re.search(r"<draft\s*>(.*?)</draft\s*>", text, re.DOTALL | re.IGNORECASE)
        if m:
            draft = m.group(1).strip()

        if not draft:
            raise OutputFormatError("未找到 <draft> 标签或内容为空")
        return thought, draft

    def _parse_thought_only(self, text: str) -> str:
        """仅从推理阶段输出中提取 <thought_process>，若无标签则返回整段文本。"""
        text = (text or "").strip()
        m = re.search(r"<thought_process\s*>(.*?)</thought_process\s*>", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(r"<thought\s*>(.*?)</thought\s*>", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return text

    def _build_draft_system_static(self, style_guide: Optional[Any] = None) -> str:
        """拼接文笔阶段静态 system：人设 + 风格/篇幅规则，便于缓存。"""
        base = DRAFT_SYSTEM_PROMPT
        if not style_guide:
            return base
        parts = [base, "\n## 文笔与篇幅要求（本任务固定）"]
        if getattr(style_guide, "pacing_rules", None):
            parts.append(f"- 行文节奏: {style_guide.pacing_rules}")
        if getattr(style_guide, "dialogue_style", None):
            parts.append(f"- 对话风格: {style_guide.dialogue_style}")
        if getattr(style_guide, "vocabulary_features", None) and style_guide.vocabulary_features:
            parts.append("- 可参考词汇/句式: " + "；".join(style_guide.vocabulary_features[:15]))
        if getattr(style_guide, "avg_chapter_length", None) and style_guide.avg_chapter_length:
            parts.append(f"- 参考书平均每章约 {int(style_guide.avg_chapter_length)} 字，正文字数须贴合指令中的「约 N 字」。")
        return "\n".join(parts)

    async def generate_chapter(
        self,
        request: WriteRequest,
        book_id: Optional[str] = None,
        *,
        style_guide: Optional[Any] = None,
    ) -> WriteResponse:
        """
        核心生成：组装上下文 → 发 LLM（推理+文笔）→ 解析 CoT 与正文 → 审查 → 沉淀。
        为利于缓存：推理/文笔的 system 放静态人设与风格，user 放当前 RAG+指令。
        """
        context_yaml = await self._assemble_context(request, book_id)
        # 动态内容仅放 user，便于 API 层缓存 system
        user_content = f"""## RAG 上下文（精简）

```yaml
{context_yaml}
```

## 用户指令

{request.user_instruction}

请仅在 <thought_process> 标签内完成感官推演、动机博弈与动作微操的思考，不要写正文。"""

        prompt_used = user_content
        last_error: Optional[Exception] = None
        max_retries = 3

        for attempt in range(max_retries):
            try:
                if attempt > 0 and last_error:
                    user_content = prompt_used + f"\n\n【重试说明】上一轮错误: {last_error!s}\n请务必输出 <thought_process> 与 <draft> 标签，且正文不少于 500 字。"

                # 阶段一：逻辑推理（deepseek-reasoner）→ thought_process
                raw_reasoning = await self._call_llm_reasoning(user_content)
                thought_process = self._parse_thought_only(raw_reasoning or "")

                # 阶段二：文笔生成（qwen-max），system 含 StyleGuide 静态规则
                raw_draft = await self._call_llm_draft(
                    thought_process, context_yaml, request.user_instruction, style_guide=style_guide
                )
                _, draft_content = self._parse_llm_response(raw_draft or "")

                if len(draft_content) < 500:
                    last_error = OutputFormatError(f"正文字数不足 500 字（当前 {len(draft_content)} 字）")
                    continue

                response = WriteResponse(
                    thought_process=thought_process,
                    draft_content=draft_content,
                    used_context=context_yaml,
                )
                await self._save_to_preference_dataset(response, prompt_used)
                return response
            except OutputFormatError as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                continue

        raise last_error or OutputFormatError("达到最大重试次数仍无法得到合法输出")

    async def _call_llm_reasoning(self, user_content: str) -> str:
        """阶段一：逻辑推理（DeepSeek R1）生成 thought_process。"""
        base_url, api_key, model_id = self._get_model_config(self.reasoning_model)
        if not api_key:
            raise RuntimeError(
                f"未配置推理模型 API Key（WRITING_REASONING_MODEL={self.reasoning_model}）："
                "请设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY"
            )
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError("请安装 openai: pip install openai")
        client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    async def _call_llm_draft(
        self,
        thought_process: str,
        context_yaml: str,
        user_instruction: str,
        *,
        style_guide: Optional[Any] = None,
    ) -> str:
        """阶段二：文笔生成（qwen-max）。system 含静态人设+风格，user 含当前思考/上下文/指令。"""
        base_url, api_key, model_id = self._get_model_config(self.draft_model)
        if not api_key:
            raise RuntimeError(
                f"未配置文笔模型 API Key（WRITING_DRAFT_MODEL={self.draft_model}）："
                "请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY"
            )
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError("请安装 openai: pip install openai")
        system_static = self._build_draft_system_static(style_guide)
        user_content = f"""## 思考过程（逻辑推理结果）

{thought_process}

## RAG 上下文

```yaml
{context_yaml}
```

## 用户指令

{user_instruction}

请仅在 <draft> 标签内输出不少于 1000 字的网文正文，不要输出其它内容。"""
        client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_static},
                {"role": "user", "content": user_content},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    async def _save_to_preference_dataset(self, response: WriteResponse, prompt_used: str) -> None:
        """
        将成功生成的结果追加写入 JSONL，作为未来 SFT/RLHF 微调语料。
        格式：每行一个 JSON，包含 input（完整 prompt）、thought_process、draft_content。
        """
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "input": prompt_used,
            "thought_process": response.thought_process,
            "draft_content": response.draft_content,
            "used_context": response.used_context,
        }
        with open(self.dataset_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
