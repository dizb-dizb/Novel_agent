# -*- coding: utf-8 -*-
"""
逻辑主编（Logic Master）：强推理大模型驱动，不写正文，只做逻辑推演与单章设计。
确保新章节与数据库现状一致，禁止凭空捏造未定义设定或让死人复活。
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

try:
    from backend.schemas.orchestrator_models import (
        BookState,
        ChapterDesign,
        MutationPremise,
        OriginalChapterNode,
        StyleGuide,
    )
except ImportError:
    from schemas.orchestrator_models import (
        BookState,
        ChapterDesign,
        MutationPremise,
        OriginalChapterNode,
        StyleGuide,
    )

logger = logging.getLogger(__name__)

# ---------- 异常 ----------


class LogicConflictError(Exception):
    """细纲与数据库现状存在逻辑冲突（如使用未定义角色）。"""
    pass


# ---------- 提示词 ----------


STYLE_EXTRACT_SYSTEM = """你是一位专业的网文分析师。根据给定的参考书片段，提取结构化的文风特征。
请严格输出一个 JSON 对象，且仅输出该 JSON，不要用 markdown 代码块包裹。字段如下：
- reference_book_name: 参考书名（字符串）
- vocabulary_features: 高频词汇或句式特征列表（字符串数组）
- pacing_rules: 行文节奏规则描述（字符串，如「战斗描写需占三成」）
- dialogue_style: 对话风格描述（字符串）"""


REVIEW_DESIGN_SYSTEM = """你是整本仿写系统的「逻辑主编」。你的职责是：基于当前全书进度和四大数据库（角色、设定、关系、情节）的最新状态，推演下一步最合理的发展，输出下一章的细纲。

你必须：
1. 先进行 <logic_review>：审查数据库现状，说明为何这样设计、有无逻辑红线、是否出现数据库中没有的角色/设定。
2. 再输出 <chapter_design>：一个符合以下结构的 JSON 对象（且仅此 JSON，不要用 markdown 代码块包裹）。
   - chapter_number: 本章序号（整数，从 1 开始）
   - pov_character: 视角人物（必须是数据库中已有的角色名或 ID）
   - required_events: 本章必须发生的事件列表（字符串数组）
   - logic_constraints: 逻辑约束列表（字符串数组），如「主角目前重伤，不能使用高阶魔法」
   - adapted_from_chapter: 若为适配原著则填原著章节号，否则可省略或 null

禁止：凭空捏造数据库中没有的超模法宝、未出场角色、已死亡角色复活。若无法从数据库中合理推演，请在 logic_review 中说明并给出保守设计。"""


# ---------- 逆向重构：逻辑适配器（同构映射） ----------


ADAPTER_SYSTEM = """你是一个顶级的网文剧情适配专家。你的任务是将【原著章节事件】完美平移到【新世界观基调】下，绝对保持原著的叙事节奏、情绪起伏和事件功能（如：原著是打脸反派，新细纲也必须是打脸反派；原著是立三年之约，新细纲也必须是等价的情感/契约承诺）。

执行步骤（缺一不可）：

1. <logic_review>：分析原著事件在当前新数据库状态下是否成立。
   - 若有旧设定的残留（如原著的魔法，新设定是科技），或某个角色在新设定中状态不同，必须指出冲突。
   - 说明原著每个事件在新设定下的等价物是否已由 character_mapping / core_rule_changes 覆盖；若有缺口需在 logic_constraints 中写明。

2. <chapter_design>：输出适配后的新章节细纲（仅一个 JSON 对象，不要用 markdown 代码块包裹）。
   - chapter_number: 新书本章序号（与原著章节号一致或由调用方约定）
   - pov_character: 视角人物（必须使用新设定下的名字，即 character_mapping 中的「新书」名；若映射表中无则用数据库中已有的新设定角色名）
   - required_events: 将原著 original_events 逐条替换为符合新设定的动作/事件描述，因果链与原著同构。
   - logic_constraints: 逻辑约束列表，确保不违反新世界观与当前数据库状态。
   - adapted_from_chapter: 必须填写，即原著章节号（original_chapter_number）。

禁止：改变原著事件的因果顺序与功能（如把「打脸」改成「和解」）；禁止使用未在 character_mapping 或数据库中出现的角色名（新设定名）。"""


# ---------- 弧线级批量构思（一次处理 N 章，摊薄推理耗时） ----------


ADAPTER_BATCH_SYSTEM = """你是一个顶级的网文剧情适配专家。你本次收到的是【一批共 N 章】的原著大纲，请在一个思维链中统筹评估这 N 章的逻辑变异，然后输出包含 N 个元素的 JSON 数组。

执行步骤（缺一不可）：
1. <logic_review>：一次性分析本批 N 章原著事件在当前新数据库状态下是否成立；若有设定/角色冲突，指出并说明如何在新设定下等价替换。
2. <chapter_design_batch>：输出一个 JSON 数组，长度为 N，与输入的 N 个原著章节一一对应。每个元素为单章细纲对象，且仅包含以下字段（极其精炼，禁止长篇大论）：
   - chapter_number: 整数（与原著章节号一致）
   - pov_character: 视角人物（新设定下名字）
   - required_events: 字符串数组，每一条必须是短句式，不超过 30 字，仅保留核心动作/事件
   - logic_constraints: 字符串数组，每条不超过 20 字
   - adapted_from_chapter: 整数，即对应原著章节号

严格要求：required_events 禁止写成长篇大论或 800 字小作文，只写短句列表。把表达欲留给后续的编写模块。"""


# ---------- LogicMasterAgent ----------


class LogicMasterAgent:
    """系统大脑层：调用强推理模型（如 DeepSeek-R1）做风格提取与逻辑审查 + 单章设计。"""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
    ) -> None:
        self.model = (
            model
            or os.getenv("WRITING_REASONING_MODEL")
            or os.getenv("ANALYZER_HIGH_MODEL")
            or "deepseek-reasoner"
        ).strip()
        self._api_key = (api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        self._base_url = (base_url or "").strip() or None
        self.max_retries = max_retries

    def _get_client_config(self) -> tuple[Optional[str], str, str]:
        """解析推理模型配置。"""
        try:
            from src.utils.llm_client import get_model_config
            base_url, key, model_id = get_model_config(self.model)
            return base_url, (key or self._api_key), (model_id or self.model)
        except ImportError:
            pass
        base = self._base_url or "https://api.deepseek.com"
        key = self._api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        return base, key, self.model

    async def _call_llm(self, system: str, user_content: str) -> str:
        """
        异步调用推理模型。为利于 Prompt 缓存：system 放静态内容（人设+变异基调等），user 放动态内容（本章节点+快照）。
        """
        base_url, api_key, model_id = self._get_client_config()
        if not api_key:
            raise RuntimeError("未配置推理模型 API Key（DEEPSEEK_API_KEY 或 OPENAI_API_KEY）")
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError("请安装 openai: pip install openai")
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    def _build_static_system_for_adapter(self, mutation_premise: MutationPremise) -> str:
        """拼接静态 system：人设 + 变异基调，便于 API 层做 Prompt 缓存（不随章节变化）。"""
        static_premise = f"""
## 变异基调（本任务固定，用于缓存）
- 新世界观: {mutation_premise.new_world_setting or '（未指定）'}
- 角色映射（原著名 -> 新书名）: {json.dumps(mutation_premise.character_mapping, ensure_ascii=False)}
- 核心法则变动:
{chr(10).join("- " + r for r in mutation_premise.core_rule_changes) or "  - （无）"}
"""
        return (ADAPTER_SYSTEM + static_premise).strip()

    def _build_static_system_for_batch_adapter(self, mutation_premise: MutationPremise) -> str:
        """拼接批量适配的静态 system：人设 + 变异基调。"""
        static_premise = f"""
## 变异基调（本任务固定）
- 新世界观: {mutation_premise.new_world_setting or '（未指定）'}
- 角色映射: {json.dumps(mutation_premise.character_mapping, ensure_ascii=False)}
- 核心法则变动: {chr(10).join("- " + r for r in mutation_premise.core_rule_changes) or "  - （无）"}
"""
        return (ADAPTER_BATCH_SYSTEM + static_premise).strip()

    def extract_style_guide(self, reference_text: str) -> StyleGuide:
        """
        输入参考书片段，输出结构化的文风特征。建议只执行一次并缓存。
        同步包装（内部 asyncio.run）；若已在 async 上下文中请用 extract_style_guide_async。
        """
        if not reference_text or not reference_text.strip():
            return StyleGuide(reference_book_name="未知", vocabulary_features=[], pacing_rules="", dialogue_style="")
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("当前已在异步事件循环中，请使用 extract_style_guide_async(reference_text)")
        async def _run():
            return await self._call_llm(STYLE_EXTRACT_SYSTEM, f"参考书片段：\n\n{reference_text[:8000]}")
        raw = asyncio.run(_run())
        return self._parse_style_guide(raw)

    async def extract_style_guide_async(self, reference_text: str) -> StyleGuide:
        """异步版本：从参考书片段提取 StyleGuide。"""
        if not reference_text or not reference_text.strip():
            return StyleGuide(reference_book_name="未知", vocabulary_features=[], pacing_rules="", dialogue_style="")
        raw = await self._call_llm(STYLE_EXTRACT_SYSTEM, f"参考书片段：\n\n{reference_text[:8000]}")
        return self._parse_style_guide(raw)

    def _parse_style_guide(self, raw: str) -> StyleGuide:
        """从 LLM 输出中解析 JSON 并构造 StyleGuide。"""
        raw = (raw or "").strip()
        # 去掉可能的 markdown 代码块
        for prefix in ("```json", "```"):
            if raw.startswith(prefix):
                raw = raw[len(prefix) :].strip()
            if raw.endswith("```"):
                raw = raw[:-3].strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning("StyleGuide JSON 解析失败，使用默认: %s", e)
            return StyleGuide(reference_book_name="未知", vocabulary_features=[], pacing_rules="", dialogue_style="")
        if not isinstance(data, dict):
            return StyleGuide(reference_book_name="未知", vocabulary_features=[], pacing_rules="", dialogue_style="")
        return StyleGuide(
            reference_book_name=data.get("reference_book_name", "未知"),
            vocabulary_features=data.get("vocabulary_features") or [],
            pacing_rules=data.get("pacing_rules", ""),
            dialogue_style=data.get("dialogue_style", ""),
        )

    async def review_and_design(
        self,
        current_state: BookState,
        db_snapshot: Dict[str, Any],
    ) -> ChapterDesign:
        """
        接收当前全书进度和四大数据库快照，先进行 logic_review，再输出符合 Pydantic 的 ChapterDesign。
        若细纲与现状存在逻辑冲突（如未定义角色），触发重试。
        """
        user_content = f"""当前全书状态：
- 已写完章节数（当前进度）: {current_state.current_chapter}
- 当前主线目标: {current_state.main_plot_goal or '（未指定）'}

四大数据库快照（请严格基于此推演，禁止捏造其中没有的角色/设定）：
```json
{json.dumps(db_snapshot, ensure_ascii=False, indent=2)}
```

请先输出 <logic_review>...</logic_review>，再输出 <chapter_design>...</chapter_design>，chapter_design 内仅放一个 JSON 对象。"""

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                raw = await self._call_llm(REVIEW_DESIGN_SYSTEM, user_content)
                design = self._parse_chapter_design(raw, current_state.current_chapter + 1, default_adapted_from=None)
                self._validate_design_against_snapshot(design, db_snapshot)
                return design
            except (LogicConflictError, ValueError, json.JSONDecodeError) as e:
                last_error = e
                logger.warning("review_and_design 第 %s 轮逻辑冲突或解析失败: %s", attempt + 1, e)
                user_content += f"\n\n【重试】上一轮错误: {e!s}\n请修正细纲，确保角色与设定均来自上述数据库快照，并重新输出 <logic_review> 与 <chapter_design>。"
                continue
        raise last_error or LogicConflictError("达到最大重试次数仍无法得到无冲突的细纲")

    async def adapt_and_design(
        self,
        original_node: OriginalChapterNode,
        mutation_premise: MutationPremise,
        db_snapshot: Dict[str, Any],
        *,
        validate_against_snapshot: Optional[Dict[str, Any]] = None,
    ) -> ChapterDesign:
        """
        逻辑适配器：将原著章节节点在同构映射下适配为新设定细纲。
        db_snapshot 用于组 prompt（可传入降维快照）；校验时用 validate_against_snapshot（若提供），否则用 db_snapshot。
        """
        check_snapshot = validate_against_snapshot if validate_against_snapshot is not None else db_snapshot
        # 缓存友好：system 含人设+变异基调（静态），user 仅本章节点+快照（动态）
        system_static = self._build_static_system_for_adapter(mutation_premise)
        user_content = f"""## 原著本章节点（绝对客观情节）

- 原著章节号: {original_node.chapter_number}
- 原著视角人物: {original_node.original_pov or '（未指定）'}
- 原著本章事件（按顺序）:
{chr(10).join("- " + e for e in original_node.original_events) or "  - （无）"}
- 原著本章爽点/目标: {original_node.original_goal or '（未指定）'}

## 当前新书数据库快照（请严格基于此校验，禁止捏造不存在的角色/设定）

```json
{json.dumps(db_snapshot, ensure_ascii=False, indent=2)}
```

请先输出 <logic_review>...</logic_review>，再输出 <chapter_design>...</chapter_design>，chapter_design 内仅放一个 JSON 对象。"""

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                raw = await self._call_llm(system_static, user_content)
                design = self._parse_chapter_design(
                    raw,
                    expected_chapter=original_node.chapter_number,
                    default_adapted_from=original_node.chapter_number,
                )
                design.chapter_number = original_node.chapter_number
                design.adapted_from_chapter = design.adapted_from_chapter or original_node.chapter_number
                self._validate_design_against_snapshot(design, check_snapshot, mutation_premise)
                return design
            except (LogicConflictError, ValueError, json.JSONDecodeError) as e:
                last_error = e
                logger.warning("adapt_and_design 第 %s 轮逻辑冲突或解析失败: %s", attempt + 1, e)
                user_content += f"\n\n【重试】上一轮错误: {e!s}\n请修正细纲，确保角色使用新设定名且来自映射表或数据库，并重新输出 <logic_review> 与 <chapter_design>。"
                continue
        raise last_error or LogicConflictError("达到最大重试次数仍无法得到无冲突的适配细纲")

    async def review_and_design_batch(
        self,
        original_nodes_batch: List[OriginalChapterNode],
        mutation_premise: MutationPremise,
        db_snapshot: Dict[str, Any],
        *,
        validate_against_snapshot: Optional[Dict[str, Any]] = None,
    ) -> List[ChapterDesign]:
        """
        弧线级批量构思：一次将 N 章原著节点适配为新设定细纲，摊薄大模型推理耗时。
        db_snapshot 用于组 prompt（可传入降维后的快照）；校验时使用 validate_against_snapshot（若提供），否则用 db_snapshot。
        返回的 List[ChapterDesign] 与 original_nodes_batch 一一对应（按 chapter_number）。
        """
        check_snapshot = validate_against_snapshot if validate_against_snapshot is not None else db_snapshot
        if not original_nodes_batch:
            return []
        n = len(original_nodes_batch)
        batch_lines = []
        for node in original_nodes_batch:
            batch_lines.append(
                f"- 章{node.chapter_number}: POV={node.original_pov or '（未指定）'} | 目标={node.original_goal or '（无）'} | 事件={chr(44).join(node.original_events[:5]) or '（无）'}"
            )
        # 缓存友好：system 含人设+变异基调，user 仅本批节点+快照
        system_static = self._build_static_system_for_batch_adapter(mutation_premise)
        user_content = f"""## 本批共 {n} 章原著节点（按顺序）

{chr(10).join(batch_lines)}

## 当前新书数据库快照（请严格基于此校验）

```json
{json.dumps(db_snapshot, ensure_ascii=False, indent=2)}
```

请先输出 <logic_review>...</logic_review>，再输出 <chapter_design_batch>...</chapter_design_batch>，其中仅放一个 JSON 数组，长度为 {n}，每个元素为单章细纲对象。"""

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                raw = await self._call_llm(system_static, user_content)
                designs = self._parse_batch_chapter_designs(
                    raw, original_nodes_batch
                )
                if len(designs) != n:
                    raise ValueError(
                        f"批量输出章数不符：期望 {n} 章，得到 {len(designs)} 章（可能 JSON 被截断）"
                    )
                for i, (d, orig) in enumerate(zip(designs, original_nodes_batch)):
                    d.chapter_number = orig.chapter_number
                    d.adapted_from_chapter = d.adapted_from_chapter or orig.chapter_number
                    self._validate_design_against_snapshot(d, check_snapshot, mutation_premise)
                return designs
            except (LogicConflictError, ValueError, json.JSONDecodeError) as e:
                last_error = e
                logger.warning(
                    "review_and_design_batch 第 %s 轮失败（章数不符或解析错误）: %s",
                    attempt + 1,
                    e,
                )
                user_content += f"\n\n【重试】上一轮错误: {e!s}\n请确保输出一个长度为 {n} 的 JSON 数组，且每个元素包含 chapter_number/pov_character/required_events/logic_constraints/adapted_from_chapter。"
                continue
        raise last_error or LogicConflictError(
            "达到最大重试次数仍无法得到完整且无冲突的批量适配细纲"
        )

    def _parse_batch_chapter_designs(
        self,
        raw: str,
        original_nodes_batch: List[OriginalChapterNode],
    ) -> List[ChapterDesign]:
        """从 LLM 输出中提取 <chapter_design_batch> 内的 JSON 数组并解析为 List[ChapterDesign]。"""
        raw = (raw or "").strip()
        m = re.search(
            r"<chapter_design_batch\s*>(.*?)</chapter_design_batch\s*>",
            raw,
            re.DOTALL | re.IGNORECASE,
        )
        if not m:
            raise ValueError("未找到 <chapter_design_batch> 标签")
        json_str = m.group(1).strip()
        for prefix in ("```json", "```"):
            if json_str.startswith(prefix):
                json_str = json_str[len(prefix) :].strip()
            if json_str.endswith("```"):
                json_str = json_str[:-3].strip()
        arr = json.loads(json_str)
        if not isinstance(arr, list):
            raise ValueError("chapter_design_batch 内容应为 JSON 数组")
        designs = []
        for i, item in enumerate(arr):
            if not isinstance(item, dict):
                raise ValueError(f"第 {i + 1} 个元素不是对象")
            ch = item.get("chapter_number")
            orig = original_nodes_batch[i] if i < len(original_nodes_batch) else None
            expected_ch = orig.chapter_number if orig else (i + 1)
            if ch is not None and not isinstance(ch, int):
                try:
                    ch = int(ch)
                except (TypeError, ValueError):
                    ch = expected_ch
            adapted = item.get("adapted_from_chapter")
            if adapted is not None and not isinstance(adapted, int):
                try:
                    adapted = int(adapted)
                except (TypeError, ValueError):
                    adapted = expected_ch
            designs.append(
                ChapterDesign(
                    chapter_number=ch or expected_ch,
                    pov_character=(item.get("pov_character") or "").strip(),
                    required_events=item.get("required_events") or [],
                    logic_constraints=item.get("logic_constraints") or [],
                    adapted_from_chapter=adapted if adapted is not None else expected_ch,
                )
            )
        return designs

    def _parse_chapter_design(
        self,
        raw: str,
        expected_chapter: int,
        default_adapted_from: Optional[int] = None,
    ) -> ChapterDesign:
        """从 LLM 输出中提取 <chapter_design> 内的 JSON 并解析为 ChapterDesign。"""
        raw = (raw or "").strip()
        m = re.search(r"<chapter_design\s*>(.*?)</chapter_design\s*>", raw, re.DOTALL | re.IGNORECASE)
        if not m:
            raise ValueError("未找到 <chapter_design> 标签")
        json_str = m.group(1).strip()
        for prefix in ("```json", "```"):
            if json_str.startswith(prefix):
                json_str = json_str[len(prefix) :].strip()
            if json_str.endswith("```"):
                json_str = json_str[:-3].strip()
        data = json.loads(json_str)
        adapted = data.get("adapted_from_chapter")
        if adapted is not None and not isinstance(adapted, int):
            try:
                adapted = int(adapted)
            except (TypeError, ValueError):
                adapted = default_adapted_from
        design = ChapterDesign(
            chapter_number=data.get("chapter_number", expected_chapter),
            pov_character=(data.get("pov_character") or "").strip(),
            required_events=data.get("required_events") or [],
            logic_constraints=data.get("logic_constraints") or [],
            adapted_from_chapter=adapted if adapted is not None else default_adapted_from,
        )
        return design

    def _validate_design_against_snapshot(
        self,
        design: ChapterDesign,
        db_snapshot: Dict[str, Any],
        mutation_premise: Optional[MutationPremise] = None,
    ) -> None:
        """校验细纲：视角人物等是否在数据库或映射表中存在，否则抛出 LogicConflictError。"""
        characters = set()
        for key in ("characters", "角色", "人物", "entities_by_type"):
            val = db_snapshot.get(key)
            if isinstance(val, list):
                for c in val:
                    if isinstance(c, dict):
                        name = (c.get("name") or c.get("id") or "").strip()
                        if name:
                            characters.add(name)
                    elif isinstance(c, str):
                        characters.add(c.strip())
            elif isinstance(val, dict):
                for k, v in (val or {}).items():
                    if isinstance(v, list):
                        for c in v:
                            if isinstance(c, dict):
                                name = (c.get("name") or c.get("id") or "").strip()
                                if name:
                                    characters.add(name)
                            elif isinstance(c, str):
                                characters.add(c.strip())
        if mutation_premise and mutation_premise.character_mapping:
            for new_name in mutation_premise.character_mapping.values():
                if isinstance(new_name, str) and new_name.strip():
                    characters.add(new_name.strip())
        # 当快照中角色很少（如分析仅写入 2 个）时，放宽校验：允许视角人物来自原著正文，不报错
        if design.pov_character and characters and design.pov_character not in characters:
            if len(characters) <= 5:
                logger.debug(
                    "快照仅 %s 个角色，放宽校验：允许视角人物「%s」来自原著",
                    len(characters),
                    design.pov_character,
                )
            else:
                raise LogicConflictError(f"视角人物「{design.pov_character}」不在数据库角色列表或角色映射表中")

    # ---------- 因果修复：单节点逻辑审查（MutationPropagator 调用） ----------

    NODE_REVIEW_SYSTEM = """你是逻辑主编。上游剧情已发生变异。请对比「当前章节细纲」与「最新数据库状态」。
若当前章节使用了已死亡的角色、不存在的法宝、或动机不再成立，请【重写该章细纲】以符合新逻辑，同时保留原有行文节奏。
若逻辑自洽，则无需修改。
请严格按以下 JSON 输出（仅此 JSON，不要 markdown 包裹）：
{
  "should_rewrite": true 或 false,
  "new_summary": "若需重写则给出新细纲全文，否则为 null",
  "logic_notes": "审查说明：为何一致或为何需改"
}"""

    async def review_node_logic(
        self,
        request: "NodeReviewRequest",
    ) -> "NodeReviewResult":
        """
        节点逻辑审查：上游变异后，判断当前节点细纲是否仍自洽；若否则返回新细纲供覆写。
        """
        try:
            from backend.schemas.plot_schemas import NodeReviewRequest, NodeReviewResult
        except ImportError:
            from schemas.plot_schemas import NodeReviewRequest, NodeReviewResult

        user_content = f"""当前节点 ID: {request.node_id}
章节序号: {request.chapter_index}

当前章节细纲：
{request.current_summary or '（无）'}

涉及角色: {request.involved_characters or []}

上游变异摘要：
{request.upstream_mutation_summary or '（无）'}

最新数据库快照（角色/设定/情节）：
```json
{json.dumps(request.db_snapshot, ensure_ascii=False, indent=2)}
```

请判断：当前细纲是否与数据库一致？是否使用了已删除角色、不存在设定或不再成立的动机？若需重写请给出 new_summary。"""

        raw = await self._call_llm(self.NODE_REVIEW_SYSTEM, user_content)
        raw = (raw or "").strip()
        for prefix in ("```json", "```"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        data = json.loads(raw)
        return NodeReviewResult(
            should_rewrite=bool(data.get("should_rewrite")),
            new_summary=data.get("new_summary") if data.get("should_rewrite") else None,
            logic_notes=str(data.get("logic_notes") or "").strip(),
        )
