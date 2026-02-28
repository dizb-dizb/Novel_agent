# -*- coding: utf-8 -*-
"""
用户心理专家（User Profiler Agent）：锚定用户爽点，转化为写作指令。
与分析、生成模块并列，提供偏好种子、情感续写导航、类似体验迁移与满意度反馈闭环。
"""
from .schema import (
    UserProfile,
    UserPreferenceProtocol,
    UserContext,
    TropePreference,
    ParsedIntent,
    FeedbackRecord,
    TropeCard,
    WorldRuleCard,
)
from .psychology_expert import (
    build_profile_from_dialogue,
    build_protocol_from_profile,
    get_preference_patch,
    load_preference_protocol,
    save_preference_protocol,
    narrative_steering_question,
    produce_user_context,
)
from .intent_analyzer import parse_intent, extract_continuation_intent_for_writer
from .satisfaction_tracker import SatisfactionTracker, update_profile_from_feedback
from .recommendation import (
    extract_trope_card,
    extract_world_rule_card,
    suggest_similar_setting,
)

__all__ = [
    "UserProfile",
    "UserPreferenceProtocol",
    "UserContext",
    "TropePreference",
    "ParsedIntent",
    "FeedbackRecord",
    "TropeCard",
    "WorldRuleCard",
    "build_profile_from_dialogue",
    "build_protocol_from_profile",
    "get_preference_patch",
    "load_preference_protocol",
    "save_preference_protocol",
    "narrative_steering_question",
    "produce_user_context",
    "parse_intent",
    "extract_continuation_intent_for_writer",
    "SatisfactionTracker",
    "update_profile_from_feedback",
    "extract_trope_card",
    "extract_world_rule_card",
    "suggest_similar_setting",
]
