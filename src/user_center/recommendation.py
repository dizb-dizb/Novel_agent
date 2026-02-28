# -*- coding: utf-8 -*-
"""
类似体验迁移（Recommendation）：根据用户对原著的喜爱，推荐/构思「同类体验」的剧本。
提取原书 WorldRuleCard 与 TropeCard，换皮（如修仙→科幻）保留人物性格内核与爽点结构。
"""
from typing import Any, Dict, List, Optional

from .schema import TropeCard, UserProfile, WorldRuleCard

try:
    from src.analyzer.state_schema import AnalysisState, MetaProtocol
except ImportError:
    AnalysisState = None  # type: ignore
    MetaProtocol = None  # type: ignore


def extract_trope_card(
    analysis_state: Optional["AnalysisState"] = None,
    book_id: str = "",
    plot_summaries: Optional[List[str]] = None,
) -> TropeCard:
    """
    从分析状态或剧情摘要中提取原书「爽点结构」卡片。
    供换皮时保留精神内核。
    """
    card = TropeCard(book_id=book_id)
    if not analysis_state and not plot_summaries:
        return card
    if analysis_state:
        # 从节点类型与摘要中简单归纳
        types_seen: Dict[str, int] = {}
        kernels: List[str] = []
        for n in (analysis_state.plot_tree or {}).values():
            t = n.type or "event"
            types_seen[t] = types_seen.get(t, 0) + 1
            if n.summary and len(kernels) < 10:
                kernels.append(n.summary[:80])
        # 常见类型映射到爽点标签
        trope_map = {
            "battle": "绝境反杀/战斗",
            "power_up": "升级/突破",
            "revenge": "复仇/打脸",
            "romance": "感情/互动",
            "scheme": "智斗/谋略",
            "daily": "日常/种田",
        }
        for t, c in sorted(types_seen.items(), key=lambda x: -x[1])[:6]:
            for k, tag in trope_map.items():
                if k in t.lower() or (tag and tag.split("/")[0] in t):
                    card.tropes.append(tag.split("/")[0])
                    break
            else:
                card.tropes.append(t)
        card.character_kernels = kernels[:5]
        card.pacing = "medium"
    if plot_summaries:
        card.summary = "；".join(plot_summaries[:3])[:300]
    card.tropes = list(dict.fromkeys(card.tropes))[:10]
    return card


def extract_world_rule_card(
    analysis_state: Optional["AnalysisState"] = None,
    book_id: str = "",
) -> WorldRuleCard:
    """从 AnalysisState 的元协议与卡片中提取世界观/规则摘要，换皮时保留逻辑。"""
    card = WorldRuleCard(book_id=book_id)
    if not analysis_state:
        return card
    meta = getattr(analysis_state, "meta_protocol", None) or None
    if meta:
        for r in getattr(meta, "logic_red_lines", []) or []:
            card.rules.append(f"[{getattr(r, 'category', '')}] {getattr(r, 'rule', '')}")
        card.term_mapping = dict(getattr(meta, "term_mapping", None) or {})
    for c in getattr(analysis_state, "cards", []) or []:
        if getattr(c, "type", "") == "设定" and len(card.rules) < 15:
            card.rules.append(getattr(c, "description", "")[:120])
    card.genre = _infer_genre_from_rules(card.rules)
    return card


def _infer_genre_from_rules(rules: List[str]) -> str:
    """从规则描述推断题材。"""
    text = " ".join(rules).lower()
    if "境界" in text or "修炼" in text or "灵气" in text:
        return "修仙"
    if "机甲" in text or "星际" in text or "科技" in text:
        return "科幻"
    if "都市" in text or "现代" in text:
        return "都市"
    return ""


def suggest_similar_setting(
    profile: UserProfile,
    trope_card: TropeCard,
    world_card: WorldRuleCard,
    new_skin: str = "科幻",
    max_outline_len: int = 800,
) -> str:
    """
    根据用户偏好 + 原书 TropeCard/WorldRuleCard，生成「换皮」后的新书/剧本构思。
    保留人物性格内核与爽点结构，替换皮相（如修仙→科幻）。
    """
    tropes = ", ".join(trope_card.tropes[:6]) or "无"
    kernels = ", ".join(trope_card.character_kernels[:3]) or "无"
    rules = "\n".join(f"- {r}" for r in world_card.rules[:8])
    user_pref = ""
    if profile.fetish_elements:
        user_pref = f"用户偏好要素：{'、'.join(profile.fetish_elements[:5])}。"
    from src.utils.llm_client import chat_high_quality
    user = f"""请根据以下原书「精神内核」与用户偏好，构思一个「{new_skin}」题材的新故事设定（换皮）。

原书爽点结构：{tropes}
人物性格内核：{kernels}
原书世界观/规则摘要：
{rules}
{user_pref}

要求：
1. 保留原书的爽点类型与人物成长节奏，仅将背景与设定改为「{new_skin}」风格。
2. 输出 200-{max_outline_len} 字的故事背景 + 主角定位 + 前几章可用的冲突钩子。不要写正文。
"""
    msgs = [
        {"role": "system", "content": "你是网文策划，只输出新书构思摘要，不写正文。"},
        {"role": "user", "content": user},
    ]
    raw = (chat_high_quality(msgs) or "").strip()
    return raw[:max_outline_len] if raw else ""
