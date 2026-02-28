# -*- coding: utf-8 -*-
"""模块2：剧情树与知识卡片分析（元协议 + 双环提取 + 回溯查询）。"""
from .models import KnowledgeCard, PlotNode
from .extractor import extract_cards_from_chapter, extract_cards_from_window
from .plot_tree import add_node, get_children
from .state_schema import (
    AnalysisState,
    ConflictMark,
    ElementFieldDef,
    LogicRedLine,
    MetaProtocol,
    NovelDatabase,
    PendingPatch,
    UNCLASSIFIED_FIELD_NAME,
    build_novel_database,
    create_initial_state,
    load_state,
    save_state,
)
from .sampler import smart_sample
from .protocol_generator import generate_meta_protocol
from .refiner import detect_conflicts_and_merge
from .backtrack import check_consistency, query_causal_parents
from .pipeline import (
    run_phase1_sampling_and_protocol,
    run_phase2_concurrent_extract_then_consolidate,
    run_phase2_incremental_extract,
    run_phase2_per_chapter_then_consolidate,
    run_full_pipeline,
    run_phase2_only,
)
from .causal_tracker import (
    assess_rewrite_impact,
    RewriteImpactReport,
    get_affected_nodes_after_rewrite,
)
from .double_check import logic_check, style_check, double_check_gate
from .rewrite_pipeline import (
    step1_impact_assessment,
    step2_dynamic_context,
    step3_double_check,
    step4_incremental_sync,
    run_rewrite_flow,
    load_state_for_rewrite,
)

__all__ = [
    "PlotNode",
    "KnowledgeCard",
    "extract_cards_from_chapter",
    "extract_cards_from_window",
    "add_node",
    "get_children",
    "AnalysisState",
    "MetaProtocol",
    "LogicRedLine",
    "ElementFieldDef",
    "ConflictMark",
    "PendingPatch",
    "UNCLASSIFIED_FIELD_NAME",
    "NovelDatabase",
    "build_novel_database",
    "create_initial_state",
    "save_state",
    "load_state",
    "smart_sample",
    "generate_meta_protocol",
    "detect_conflicts_and_merge",
    "check_consistency",
    "query_causal_parents",
    "run_phase1_sampling_and_protocol",
    "run_phase2_incremental_extract",
    "run_phase2_per_chapter_then_consolidate",
    "run_phase2_concurrent_extract_then_consolidate",
    "run_full_pipeline",
    "run_phase2_only",
    "assess_rewrite_impact",
    "RewriteImpactReport",
    "get_affected_nodes_after_rewrite",
    "logic_check",
    "style_check",
    "double_check_gate",
    "step1_impact_assessment",
    "step2_dynamic_context",
    "step3_double_check",
    "step4_incremental_sync",
    "run_rewrite_flow",
    "load_state_for_rewrite",
]
