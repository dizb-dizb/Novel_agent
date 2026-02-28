# -*- coding: utf-8 -*-
"""
智能抽样：高质量模型扫描全书目录与前 50 章摘要，按语义挑选约 10%–20% 关键节点。
章节少的书多抽一些，章节多的书少一些；抽样结果用于元协议生成。
"""
import json
from typing import List, Optional, Tuple

from src.utils.llm_client import chat_high_quality


def _sample_ratio(total_chapters: int) -> float:
    """根据总章数决定抽样比例：章少多抽，章多少抽，约 10%–20%。"""
    if total_chapters <= 30:
        return 0.20
    if total_chapters <= 100:
        return 0.15
    return 0.10


def _build_toc_and_preview(
    chapters: List[dict],
    max_preview_chapters: int = 50,
    max_content_len: int = 300,
) -> str:
    """构建目录与前若干章的摘要预览文本，供 LLM 阅读。"""
    lines = ["## 全书目录（章节标题）", ""]
    for i, ch in enumerate(chapters):
        title = (ch.get("chapter_title") or "").strip()
        cid = ch.get("chapter_id", "")
        lines.append(f"  {i + 1}. [{cid}] {title}")
    lines.append("")
    lines.append("## 前若干章内容摘要（供识别关键节点）")
    preview = chapters[:max_preview_chapters]
    for i, ch in enumerate(preview):
        title = (ch.get("chapter_title") or "").strip()
        content = (ch.get("content") or "").strip()
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        lines.append(f"### 第 {i + 1} 章 {title}")
        lines.append(content or "(无正文)")
        lines.append("")
    return "\n".join(lines)


def smart_sample(
    chapters: List[dict],
    total_chapter_count: Optional[int] = None,
) -> Tuple[List[int], List[str]]:
    """
    使用高质量模型从目录与前 50 章摘要中挑选关键章节索引。
    :param chapters: 列表，每项含 chapter_id, chapter_title, content（可选）
    :param total_chapter_count: 若传入则用该书总章数；否则 len(chapters)
    :return: (选中的章节索引 0-based, 选中的 chapter_id 列表)
    """
    total = total_chapter_count if total_chapter_count is not None else len(chapters)
    ratio = _sample_ratio(total)
    n_select = max(1, int(total * ratio))

    toc_preview = _build_toc_and_preview(chapters)

    user = f"""请基于以下全书目录与前若干章内容摘要，从整本书中选出约 {n_select} 个「关键节点」章节。
关键节点包括：力量体系设定点、重大势力更迭点、核心伏笔埋设点、世界观扩展点等（信息高密度章节）。
总章数约 {total}，请选出约 {int(ratio * 100)}% 即约 {n_select} 章。

要求：
1. 只返回一个 JSON 数组，数组元素为章节序号（从 1 开始），例如 [1, 5, 12, 20, ...]。
2. 不要返回其他解释，仅此 JSON 数组。

{toc_preview}
"""

    messages = [
        {"role": "system", "content": "你是一个小说分析助手，负责从目录与摘要中识别关键剧情节点章节，只输出 JSON 数组。"},
        {"role": "user", "content": user},
    ]
    raw = chat_high_quality(messages)
    indices_1based: List[int] = []
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            for p in ("```json", "```"):
                if raw.startswith(p):
                    raw = raw[len(p):].strip()
                    break
            if raw.endswith("```"):
                raw = raw[:-3].strip()
        indices_1based = json.loads(raw)
        if not isinstance(indices_1based, list):
            indices_1based = []
        indices_1based = [int(x) for x in indices_1based if isinstance(x, (int, float))]
    except (json.JSONDecodeError, ValueError):
        pass

    # 去重、排序、限制在有效范围，转为 0-based
    indices_1based = sorted(set(indices_1based))
    indices_1based = [i for i in indices_1based if 1 <= i <= total]
    if not indices_1based and total > 0:
        indices_1based = [1]
    indices_0based = [i - 1 for i in indices_1based]
    selected_ids = []
    for i in indices_0based:
        if 0 <= i < len(chapters):
            selected_ids.append(chapters[i].get("chapter_id") or str(i))
    return indices_0based, selected_ids
