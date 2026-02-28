# -*- coding: utf-8 -*-
"""
元知识模板设计：由长上下文 AI 或高质量 Agent 基于整书智能采样完成。
产出逻辑红线、要素模版（知识卡片字段）、术语映射，供低质量模型逐章提取使用。
解析失败时按 LangChain 风格重试：将错误原因反馈给模型，请求重写以保证元知识模板正确构造。
"""
import json
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from src.utils.llm_client import chat_high_quality, chat_long_context

from .extractor import _extract_json_from_response
from .state_schema import (
    ElementFieldDef,
    LogicRedLine,
    MetaProtocol,
    UNCLASSIFIED_FIELD_NAME,
)

# 元协议生成最大重试次数（首次 + 重试，共 max_retries + 1 次调用）
META_PROTOCOL_MAX_RETRIES = 2
# 首轮生成成功后，按「每轮 3 章」遍历智能采样章节做多轮优化；轮数 = 采样章数//3，且不超过此上限
MAX_REFINEMENT_ROUNDS = 10
# 优化轮每轮单次请求的「当前协议+本章节」总字符上限，避免高质量模型 128k 上下文超限（约 128k token）
MAX_REFINEMENT_USER_CHARS = 90000


def _normalize_protocol_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """兼容模型返回的中文键名，统一为英文键。"""
    if not data or not isinstance(data, dict):
        return data or {}
    out = {}
    for eng_key, aliases in [
        ("logic_red_lines", ["logic_red_lines", "逻辑红线"]),
        ("element_template", ["element_template", "要素模版", "要素模板"]),
        ("term_mapping", ["term_mapping", "术语映射"]),
        ("note", ["note", "备注"]),
        ("core_templates", ["core_templates", "核心模板"]),
    ]:
        for alias in aliases:
            if alias in data and data[alias] is not None:
                out[eng_key] = data[alias]
                break
    return out if out else data


def _parse_and_validate_protocol(raw: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    解析并校验元协议 JSON。返回 (data, error_reason)。
    若解析成功且结构可用，data 为 dict、error_reason 为空；否则 data 为 None，error_reason 为可反馈给模型的原因。
    """
    if not raw or not raw.strip():
        return None, "输出为空，请返回一个完整的 JSON 对象。"
    data = _extract_json_from_response(raw)
    if data and isinstance(data, dict):
        data = _normalize_protocol_keys(data)
    if not data or not isinstance(data, dict):
        return None, "无法从回复中解析出合法 JSON，请只输出一个 JSON 对象，不要使用 markdown 代码块或前后附加文字。"
    if not isinstance(data.get("logic_red_lines"), list) and data.get("logic_red_lines") is not None:
        return None, "logic_red_lines 必须是数组，例如 [] 或 [{ \"category\": \"...\", \"rule\": \"...\" }]。"
    if not isinstance(data.get("element_template"), list) and data.get("element_template") is not None:
        return None, "element_template 必须是数组，至少包含 name、kind、description 的字段定义。"
    if not isinstance(data.get("term_mapping"), dict) and data.get("term_mapping") is not None:
        return None, "term_mapping 必须是对象，例如 { \"别名\": \"规范名\" }。"
    if not isinstance(data.get("core_templates"), dict) and data.get("core_templates") is not None:
        return None, "core_templates 必须是对象。"
    return data, ""


# 元协议生成时采样内容总长度上限，避免超长输入导致 API 返回空或超时（约 2 万 token）
MAX_SAMPLED_CONTENT_CHARS = 70000


def _build_sampled_content(
    chapters: List[dict],
    indices_0based: List[int],
    max_chars_per_chapter: int = 8000,
    max_total_chars: Optional[int] = None,
) -> str:
    """将采样章节拼成供 LLM 阅读的文本。超过 max_total_chars 时截断，避免长上下文 API 返回空。"""
    if max_total_chars is None:
        max_total_chars = MAX_SAMPLED_CONTENT_CHARS
    parts: List[str] = []
    total_so_far = 0
    sep = "\n\n---\n\n"
    for idx in indices_0based:
        if idx < 0 or idx >= len(chapters):
            continue
        ch = chapters[idx]
        title = ch.get("chapter_title") or ""
        cid = ch.get("chapter_id") or ""
        content = (ch.get("content") or "")[:max_chars_per_chapter]
        block = f"## 第 {idx + 1} 章 [{cid}] {title}\n\n{content}"
        add_len = len(block) + (len(sep) if total_so_far else 0)
        if total_so_far + add_len > max_total_chars and parts:
            remain = max_total_chars - total_so_far - len(sep) - 80
            if remain > 500:
                block = f"## 第 {idx + 1} 章 [{cid}] {title}\n\n{content[:remain]}..."
            parts.append(block)
            break
        parts.append(block)
        total_so_far += add_len
    return sep.join(parts)


def generate_meta_protocol(
    book_id: str,
    book_title: str,
    chapters: List[dict],
    sampled_indices_0based: List[int],
    use_long_context: bool = True,
) -> MetaProtocol:
    """
    根据采样章节生成元协议。
    :param book_id: 书籍 id
    :param book_title: 书名
    :param chapters: 全书章节列表
    :param sampled_indices_0based: 采样章节的 0-based 索引
    :param use_long_context: True 时使用长上下文模型（如 128k），可带入更多采样内容，完成元知识模板设计。
    :return: MetaProtocol
    """
    max_chars = 12000 if use_long_context else 8000
    content = _build_sampled_content(
        chapters, sampled_indices_0based,
        max_chars_per_chapter=max_chars,
        max_total_chars=MAX_SAMPLED_CONTENT_CHARS,
    )

    def _make_user_message(sampled_content: str) -> str:
        return f"""请基于以下「关键节点」章节内容，为这本小说设计一份「元知识模板」JSON，供低质量模型对每一章进行快速、规范的提取。模板将用于：设定、道具、场景、角色等核心实体以及情节节点的统一抽取。

书名：{book_title}
book_id：{book_id}

## 采样章节内容

{sampled_content}

---

请严格输出一个 JSON 对象，且仅此 JSON，不要 markdown 代码块包裹以外的内容。格式如下：

{{
  "logic_red_lines": [
    {{ "category": "力量体系", "rule": "具体不可违反的规则描述", "source_chapter_ids": ["章节id"] }},
    ...
  ],
  "element_template": [
    {{ "name": "字段名", "kind": "str|list|dict|number", "description": "提取时的指导说明" }},
    ...（至少包含：名称、描述、首次出现章节；若为修仙/玄幻可增加境界、势力、法宝等）
  ],
  "term_mapping": {{
    "非规范写法1": "规范名1",
    "非规范写法2": "规范名2"
  }},
  "note": "可选备注",
  "core_templates": {{
    "character": "针对角色的提取策略说明（如何识别与描述人物、关系、立场等）",
    "setting": "针对设定/世界观的提取策略说明（力量体系、势力、规则等）",
    "item_scene": "针对道具与场景的提取策略说明（物品、地点、环境等）",
    "plot_event": "针对情节与事件的提取策略说明（因果、转折、状态变更等）"
  }}
}}

要求：
- logic_red_lines：明确不可违反的设定（如冷却时间、角色底线、世界观硬规则）。
- element_template：针对本书类型的知识卡片字段定义。
- term_mapping：书中特有名词的规范写法，避免后续理解偏差。
- core_templates：四大核心提取策略（character/setting/item_scene/plot_event），每项一段话，指导低质量模型按策略逐章提取。
"""

    user = _make_user_message(content)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "你是小说分析专家，负责设计元知识模板（逻辑红线、知识卡片字段、术语规范），输出严格符合给定 schema 的 JSON，不输出其他文字。"},
        {"role": "user", "content": user},
    ]
    chat_fn = chat_long_context if use_long_context else chat_high_quality
    last_raw = ""
    last_error = ""

    for attempt in range(META_PROTOCOL_MAX_RETRIES + 1):
        raw = chat_fn(messages)
        raw = (raw or "").strip()
        last_raw = raw
        # 长上下文返回空时，下次用高质量模型 + 更短内容重试
        if not raw and attempt < META_PROTOCOL_MAX_RETRIES and use_long_context:
            try:
                from src.utils import get_logger
                log = get_logger()
                log.info("元协议长上下文返回空，改用高质量模型并缩短采样内容重试")
                log.info("可能原因：ANALYZER_LONG_CONTEXT_MODEL 未配置或 Key 无效、请求超时、输入超长；请检查 .env 中长上下文模型与 API Key")
            except Exception:
                pass
            short_content = _build_sampled_content(
                chapters, sampled_indices_0based[:20],
                max_chars_per_chapter=4000,
                max_total_chars=40000,
            )
            messages = [
                {"role": "system", "content": "你是小说分析专家，负责设计元知识模板（逻辑红线、知识卡片字段、术语规范），输出严格符合给定 schema 的 JSON，不输出其他文字。"},
                {"role": "user", "content": _make_user_message(short_content)},
            ]
            use_long_context = False
            chat_fn = chat_high_quality
            continue
        data, error_reason = _parse_and_validate_protocol(raw)
        if data and not error_reason:
            # 首轮生成成功：高质量模型不断读取「每轮 3 章」智能采样章节，多轮优化修改元协议，直至覆盖全部采样
            n_sampled = len(sampled_indices_0based)
            refinement_rounds = min(max(0, (n_sampled + 2) // 3), MAX_REFINEMENT_ROUNDS)
            ref_iter = tqdm(range(refinement_rounds), desc="元协议优化", unit="轮", ncols=100) if tqdm else range(refinement_rounds)
            for ref_round in ref_iter:
                start = ref_round * 3
                indices_3 = sampled_indices_0based[start : start + 3]
                if not indices_3:
                    break
                three_chapter_content = _build_sampled_content(
                    chapters, indices_3,
                    max_chars_per_chapter=6000,
                    max_total_chars=25000,
                )
                try:
                    from src.utils import get_logger
                    get_logger().info("元协议优化第 %s/%s 轮，本轮参考 %s 章（智能采样）", ref_round + 1, refinement_rounds, len(indices_3))
                except Exception:
                    pass
                # 每轮使用「当前协议 + 本轮 3 章」的固定上下文，避免累积历史导致超过 128k token
                protocol_json = json.dumps(data, ensure_ascii=False, indent=2)
                if len(protocol_json) > MAX_REFINEMENT_USER_CHARS - 20000:
                    protocol_json = protocol_json[: MAX_REFINEMENT_USER_CHARS - 20000] + "\n..."
                refinement_user = f"""以下为当前元协议（JSON）。请结合本轮提供的 {len(indices_3)} 章内容，优化并输出完整的新元协议 JSON。要求：补充遗漏项、合并重复、确保 term_mapping 覆盖本章关键名词。只输出优化后的完整 JSON，不要 markdown 包裹与多余说明。

## 当前元协议

{protocol_json}

## 本轮参考章节

{three_chapter_content}
"""
                if len(refinement_user) > MAX_REFINEMENT_USER_CHARS:
                    refinement_user = refinement_user[:MAX_REFINEMENT_USER_CHARS] + "\n\n[内容已截断]"
                refinement_messages: List[Dict[str, str]] = [
                    {"role": "system", "content": "你是小说分析专家，负责优化元知识模板（逻辑红线、知识卡片字段、术语规范），输出严格符合给定 schema 的 JSON，不输出其他文字。"},
                    {"role": "user", "content": refinement_user},
                ]
                # 优化轮固定使用高质量模型（ANALYZER_HIGH_MODEL，如 DeepSeek），与首轮长上下文/高质量无关
                raw = chat_high_quality(refinement_messages)
                raw = (raw or "").strip()
                ref_data, ref_err = _parse_and_validate_protocol(raw)
                if ref_data and not ref_err:
                    data = ref_data
                else:
                    break
            break
        last_error = error_reason or "输出格式不符合元知识模板 schema。"
        if attempt < META_PROTOCOL_MAX_RETRIES:
            messages.append({"role": "assistant", "content": (raw or " ")[:8000]})
            messages.append({
                "role": "user",
                "content": f"【重试】上一轮输出解析失败。原因：{last_error}\n请严格只输出一个合法 JSON 对象，包含 logic_red_lines（数组）、element_template（数组）、term_mapping（对象）、note（可选），不要 markdown 包裹，不要多余说明。",
            })
            continue
        try:
            from src.utils import get_logger
            get_logger().warning(
                "generate_meta_protocol 重试 %s 次后仍解析失败，使用默认模版。最后错误：%s；原始回复前 500 字: %s",
                META_PROTOCOL_MAX_RETRIES + 1, last_error, (last_raw or "")[:500],
            )
        except Exception:
            pass
        return MetaProtocol(book_id=book_id, note="解析失败，使用默认模版")

    if not data or not isinstance(data, dict):
        return MetaProtocol(book_id=book_id, note="解析失败，使用默认模版")

    red_lines = []
    for item in data.get("logic_red_lines") or []:
        if isinstance(item, dict):
            red_lines.append(LogicRedLine(
                category=item.get("category", ""),
                rule=item.get("rule", ""),
                source_chapter_ids=item.get("source_chapter_ids") or [],
            ))

    template = []
    for item in data.get("element_template") or []:
        if isinstance(item, dict) and item.get("name"):
            template.append(ElementFieldDef(
                name=item.get("name", ""),
                kind=item.get("kind", "str"),
                description=item.get("description", ""),
            ))
    if not template:
        template = [
            ElementFieldDef(name="名称", kind="str", description="实体或概念名称"),
            ElementFieldDef(name="描述", kind="str", description="简要描述"),
            ElementFieldDef(name="首次出现章节", kind="str", description="chapter_id 或序号"),
        ]
    if not any(e.name == UNCLASSIFIED_FIELD_NAME for e in template):
        template.append(ElementFieldDef(
            name=UNCLASSIFIED_FIELD_NAME,
            kind="dict",
            description="提取本章出现的任何不属于已有字段的新奇设定、新地图、新概念，键值对形式。",
        ))

    term_mapping = data.get("term_mapping")
    if not isinstance(term_mapping, dict):
        term_mapping = {}

    core_templates = data.get("core_templates")
    if not isinstance(core_templates, dict):
        core_templates = {}

    return MetaProtocol(
        book_id=book_id,
        logic_red_lines=red_lines,
        element_template=template,
        term_mapping=term_mapping,
        note=data.get("note", ""),
        core_templates=core_templates,
    )
