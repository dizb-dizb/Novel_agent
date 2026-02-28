# -*- coding: utf-8 -*-
"""
用户心理专家（Psychology Expert）：分析用户偏好，生成「用户画像」与「偏好补丁」。
爽点建模 + 情感续写导航，为 Writer 提供偏好驱动的生成指令。
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.llm_client import chat_high_quality

from .schema import UserContext, UserPreferenceProtocol, UserProfile, TropePreference


def load_preference_protocol(protocol_path: Path) -> Optional[UserPreferenceProtocol]:
    """从 JSON 文件加载 User_Preference_Protocol。"""
    p = Path(protocol_path)
    if not p.is_file():
        return None
    import json
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return UserPreferenceProtocol.model_validate(data)
    except Exception:
        return None


def save_preference_protocol(protocol: UserPreferenceProtocol, protocol_path: Path) -> None:
    """将 UserPreferenceProtocol 写入 JSON。"""
    p = Path(protocol_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(protocol.model_dump_json(exclude_none=True, ensure_ascii=False), encoding="utf-8")


def build_profile_from_dialogue(
    messages: List[Dict[str, str]],
    book_id: str = "",
    user_id: str = "",
) -> UserProfile:
    """
    利用高质量模型对用户进行「心理侧写」：根据对话内容生成 UserProfile。
    messages: [{"role":"user","content":"..."}, ...]，可包含「你更喜欢快节奏还是慢热」「能接受悲剧吗」等问答。
    """
    if not messages:
        return UserProfile(book_id=book_id, user_id=user_id)
    dialogue_block = "\n".join(
        f"{m.get('role', '')}: {m.get('content', '')[:500]}"
        for m in messages[-10:]
    )
    user = f"""根据以下用户与系统的对话，推断该用户在阅读网文时的偏好，输出一个 JSON 对象（仅此 JSON，不要 markdown 包裹）。

对话摘录：
{dialogue_block}

请输出 JSON，包含以下字段（均为可选，按能推断的填）：
- preferred_pacing: "slow" | "medium" | "fast"
- tragedy_tolerance: 0~1 数字
- payoff_urgency: 0~1 数字（对「压抑后尽快爆发」的迫切程度）
- trope_preferences: [ {{ "tag": "绝境反杀", "strength": 0.8 }}, ... ]
- fetish_elements: ["智斗", "种田", ...]
- avoid_elements: ["NTR", "无脑送", ...]
- narrative_steering: 一句话描述用户当前期待的剧情走向（若对话中有提及）
"""
    msgs = [
        {"role": "system", "content": "你是用户偏好分析专家，只输出合法 JSON。"},
        {"role": "user", "content": user},
    ]
    raw = (chat_high_quality(msgs) or "").strip()
    for p in ("```json", "```"):
        if raw.startswith(p):
            raw = raw[len(p):].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    try:
        data = json.loads(raw)
    except Exception:
        return UserProfile(book_id=book_id, user_id=user_id)
    trope_prefs = [
        TropePreference(tag=x.get("tag", ""), strength=float(x.get("strength", 0.5)))
        for x in data.get("trope_preferences") or []
        if x.get("tag")
    ]
    return UserProfile(
        user_id=user_id,
        book_id=book_id,
        preferred_pacing=data.get("preferred_pacing") or "medium",
        tragedy_tolerance=float(data.get("tragedy_tolerance", 0.5)),
        payoff_urgency=float(data.get("payoff_urgency", 0.6)),
        trope_preferences=trope_prefs,
        fetish_elements=list(data.get("fetish_elements") or [])[:15],
        avoid_elements=list(data.get("avoid_elements") or [])[:15],
        narrative_steering=data.get("narrative_steering") or "",
        source="dialogue",
    )


def build_protocol_from_profile(profile: UserProfile, raw_answers: Optional[Dict[str, Any]] = None) -> UserPreferenceProtocol:
    """将 UserProfile 封装为持久化协议，可写入 User_Preference_Protocol.json。"""
    return UserPreferenceProtocol(
        user_id=profile.user_id,
        book_id=profile.book_id,
        profile=profile,
        raw_answers=raw_answers or {},
    )


def get_preference_patch(
    profile: UserProfile,
    book_id: str = "",
    chapter_index: int = -1,
    chapter_type_hint: str = "",
    max_length: int = 400,
) -> str:
    """
    生成供 Writer 注入的「偏好补丁」。
    例如：「用户极度厌恶压抑后不爆发。在本章请务必在 1500 字内安排反击。」
    """
    if not profile and not isinstance(profile, UserProfile):
        return ""
    profile = profile or UserProfile(book_id=book_id)
    parts = []
    if profile.payoff_urgency >= 0.7:
        parts.append("用户对「压抑后爆发」非常期待，请在本章内尽早安排一次情绪或实力上的释放/反击，避免长时间压抑无果。")
    elif profile.payoff_urgency <= 0.3:
        parts.append("用户可接受慢热铺垫，不必在本章强行安排高潮。")
    if profile.avoid_elements:
        parts.append(f"用户明确不希望出现以下要素：{'、'.join(profile.avoid_elements[:5])}。")
    if profile.preferred_pacing == "fast":
        parts.append("用户偏好快节奏：对话与动作可更紧凑，减少冗长环境描写。")
    elif profile.preferred_pacing == "slow":
        parts.append("用户偏好慢热：可适当增加心理描写与氛围铺垫。")
    if profile.trope_preferences:
        top = sorted(profile.trope_preferences, key=lambda x: x.strength, reverse=True)[:3]
        tags = [t.tag for t in top if t.strength >= 0.5]
        if tags:
            parts.append(f"用户偏好爽点类型：{'、'.join(tags)}，写作时可适度强化这些方向。")
    if profile.narrative_steering:
        parts.append(f"当前用户期待走向：{profile.narrative_steering}")
    patch = "；".join(parts)
    return patch[:max_length] if patch else ""


def narrative_steering_question(
    profile: UserProfile,
    situation_summary: str = "",
    options_hint: str = "",
) -> str:
    """
    情感续写导航：生成一句可向用户确认的「走向选择」问题。
    例如：「你是希望主角现在就报仇，还是希望他再隐忍一段？」
    """
    if not situation_summary:
        return ""
    user = f"""当前剧情情境摘要：{situation_summary[:500]}
{options_hint or "请根据情境，生成一句向用户确认剧情走向的提问（二选一或三选一），例如：你是希望主角现在就报仇，还是再隐忍一段？只输出这一句问话，不要解释。"}
"""
    msgs = [
        {"role": "system", "content": "你是网文续写助手，只输出一句向用户确认走向的提问。"},
        {"role": "user", "content": user},
    ]
    raw = (chat_high_quality(msgs) or "").strip()
    return raw[:300] if raw else ""


def produce_user_context(
    profile: Optional[UserProfile],
    book_id: str = "",
    chapter_index: int = -1,
    chapter_type_hint: str = "",
) -> UserContext:
    """
    一站式产出 UserContext：偏好补丁 + 走向提示 + 爽点权重。
    供生成图在「用户心理探测」节点调用。
    """
    if profile is None:
        profile = UserProfile(book_id=book_id)
    patch = get_preference_patch(profile, book_id, chapter_index, chapter_type_hint)
    trope_weights = {t.tag: t.strength for t in (profile.trope_preferences or []) if t.strength >= 0.4}
    return UserContext(
        preference_patch=patch,
        steering_hint=profile.narrative_steering or "",
        trope_weights=trope_weights,
        hard_constraints=profile.avoid_elements[:5] if profile.avoid_elements else [],
        profile_snapshot=profile,
    )
