# -*- coding: utf-8 -*-
"""
分析流水线核心流程：
  1. 整书智能采样 + 长上下文/Agent 完成元知识模板设计（高质量）
  2. 低质量模型按模板对每一章并发提取元知识块（并发快速完成）
  3. 高质量模型持续整合：设定/道具/场景/角色等实体 + 情节树 → 小说数据库
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .extractor import extract_cards_from_chapter, extract_cards_from_window
from .master_refiner import refine_after_window
from .models import KnowledgeCard, PlotNode
from .extraction_cache import get_extraction_cache
from .patch_refactor import (
    collect_patches_from_cards,
    submit_patch,
    try_refactor_if_needed,
)
from .protocol_generator import generate_meta_protocol
from .refiner import detect_conflicts_and_merge
from .sampler import smart_sample
from .state_schema import AnalysisState, create_initial_state


WINDOW_SIZE = 3


def run_phase1_sampling_and_protocol(
    book_id: str,
    title: str,
    chapters: List[dict],
    total_chapter_count: Optional[int] = None,
    use_long_context_for_protocol: bool = True,
) -> tuple[AnalysisState, List[int], List[str]]:
    """
    第一阶段：整书智能采样 + 长上下文/Agent 完成元知识模板设计（高质量模型）。
    :param use_long_context_for_protocol: 使用长上下文模型（如 128k）生成元协议，可带入更多采样内容。
    :return: (state 含 meta_protocol 与 sampled_*), sampled_indices_0based, sampled_chapter_ids
    """
    state = create_initial_state(book_id, title)
    total = total_chapter_count if total_chapter_count is not None else len(chapters)
    indices_0based, selected_ids = smart_sample(chapters, total_chapter_count=total)
    state.sampled_chapter_indices = indices_0based
    state.sampled_chapter_ids = selected_ids

    protocol = generate_meta_protocol(
        book_id, title, chapters, indices_0based,
        use_long_context=use_long_context_for_protocol,
    )
    state.meta_protocol = protocol
    return state, indices_0based, selected_ids


def run_phase2_incremental_extract(
    state: AnalysisState,
    chapters: List[dict],
    refine_every_n_windows: int = 1,
) -> AnalysisState:
    """
    第二阶段（窗口模式）：滑动窗口（3 章）低成本提取 + 高质量整合。
    外环：窗口 -> extract_cards_from_window（低成本）
    内环：refine_after_window（高质量，可每 N 窗一次）
    """
    n = len(chapters)
    window_starts = list(range(0, n, WINDOW_SIZE))
    window_iter = tqdm(window_starts, desc="窗口提取", unit="窗", ncols=100) if tqdm else window_starts
    for start in window_iter:
        window = chapters[start : start + WINDOW_SIZE]
        if not window:
            continue
        existing_ids = list(state.plot_tree.keys())
        new_cards, new_nodes = extract_cards_from_window(
            window,
            state.meta_protocol,
            existing_node_ids=existing_ids,
            window_start_index=start,
        )
        state, _ = refine_after_window(
            state,
            new_cards,
            new_nodes,
            every_n_windows=refine_every_n_windows,
            current_window_index=start // WINDOW_SIZE,
        )
        state.last_processed_chapter_index = start + len(window) - 1
    return state


def _extract_one_chapter(
    item: Tuple[int, dict, Any, str],
) -> Tuple[int, List[KnowledgeCard], List[PlotNode]]:
    """单章提取（供并发调用）。item = (chapter_index, chapter_dict, protocol, book_id)。命中缓存则直接返回。"""
    i, ch, protocol, book_id = item
    title = ch.get("chapter_title") or ""
    cid = ch.get("chapter_id") or ""
    content = (ch.get("content") or "").strip()
    if not content:
        return i, [], []
    cache = get_extraction_cache()
    cached = cache.get(book_id, cid, content, protocol)
    if cached is not None:
        return i, cached[0], cached[1]
    cards, nodes = extract_cards_from_chapter(
        chapter_title=title,
        content=content,
        chapter_id=cid,
        chapter_index=i,
        protocol=protocol,
        existing_node_ids=None,
    )
    cache.set(book_id, cid, content, protocol, cards, nodes)
    return i, cards, nodes


def run_phase2_concurrent_extract_then_consolidate(
    state: AnalysisState,
    chapters: List[dict],
    max_workers: int = 4,
) -> AnalysisState:
    """
    第二阶段（并发逐章）：低质量模型按元知识模板对每一章并发提取元知识块，再由高质量模型按章序持续整合。
    - 并发调用 extract_cards_from_chapter（低成本），快速得到各章卡片与节点
    - 按章节顺序依次 detect_conflicts_and_merge（高质量），构建情节树与知识库
    """
    protocol = state.meta_protocol
    book_id = state.book_id or ""
    n = len(chapters)
    tasks = [(i, chapters[i], protocol, book_id) for i in range(n) if (chapters[i].get("content") or "").strip()]
    num_tasks = len(tasks)
    results_by_index: dict[int, Tuple[List[KnowledgeCard], List[PlotNode]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_extract_one_chapter, t): t[0] for t in tasks}
        completed = tqdm(as_completed(futures), total=num_tasks, desc="提取章节", unit="章", ncols=100) if tqdm else as_completed(futures)
        for fut in completed:
            try:
                idx, cards, nodes = fut.result()
                results_by_index[idx] = (cards, nodes)
            except Exception:
                pass
    sorted_indices = sorted(results_by_index.keys())
    merge_iter = tqdm(sorted_indices, desc="整合章节", unit="章", ncols=100) if tqdm else sorted_indices
    for i in merge_iter:
        cards, nodes = results_by_index[i]
        state, _ = detect_conflicts_and_merge(state, cards, nodes)
        state.last_processed_chapter_index = i
        cid = (chapters[i].get("chapter_id") or "") if i < len(chapters) else ""
        for p in collect_patches_from_cards(cards, cid, i):
            submit_patch(state, p.chapter_id, p.chapter_index, p.issue, p.suggestion)
        state = try_refactor_if_needed(state, i)
    return state


def run_phase2_per_chapter_then_consolidate(
    state: AnalysisState,
    chapters: List[dict],
    consolidate_every_n_chapters: int = 1,
) -> AnalysisState:
    """
    第二阶段（逐章串行）：低质量模型按知识模板逐章提取 → 高质量模型整理整合为整本书知识库。
    - 每章调用 extract_cards_from_chapter（低成本），产出该章知识卡片与剧情节点
    - 每 N 章或每章后调用 detect_conflicts_and_merge（高质量），做冲突检测与因果树合并
    """
    protocol = state.meta_protocol
    book_id = state.book_id or ""
    cache = get_extraction_cache()
    n = len(chapters)
    chapter_iter = tqdm(range(n), desc="分析章节", unit="章", ncols=100) if tqdm else range(n)
    for i in chapter_iter:
        ch = chapters[i]
        title = ch.get("chapter_title") or ""
        cid = ch.get("chapter_id") or ""
        content = (ch.get("content") or "").strip()
        if not content:
            state.last_processed_chapter_index = i
            continue
        cached = cache.get(book_id, cid, content, protocol)
        if cached is not None:
            new_cards, new_nodes = cached
        else:
            existing_ids = list(state.plot_tree.keys())
            new_cards, new_nodes = extract_cards_from_chapter(
                chapter_title=title,
                content=content,
                chapter_id=cid,
                chapter_index=i,
                protocol=protocol,
                existing_node_ids=existing_ids,
            )
            cache.set(book_id, cid, content, protocol, new_cards, new_nodes)
        state, _ = detect_conflicts_and_merge(state, new_cards, new_nodes)
        state.last_processed_chapter_index = i
        for p in collect_patches_from_cards(new_cards, cid, i):
            submit_patch(state, p.chapter_id, p.chapter_index, p.issue, p.suggestion)
        state = try_refactor_if_needed(state, i)
    return state


def run_full_pipeline(
    book_id: str,
    title: str,
    chapters: List[dict],
    total_chapter_count: Optional[int] = None,
    refine_every_n_windows: int = 1,
    use_per_chapter_extraction: bool = True,
    use_concurrent_extraction: bool = True,
    max_concurrent_workers: int = 4,
    consolidate_every_n_chapters: int = 1,
) -> AnalysisState:
    """
    全流程：整书智能采样 → 长上下文/Agent 元知识模板 → 低质量模型逐章（可并发）提取 → 高质量模型持续整合 → 小说数据库。
    :param use_per_chapter_extraction: True 时按章提取；False 时按 3 章窗口滑动提取。
    :param use_concurrent_extraction: 逐章提取时是否并发执行（低质量模型多章并行，快速完成）。
    :param max_concurrent_workers: 并发提取的线程数。
    :param consolidate_every_n_chapters: 逐章串行模式下每 N 章做一次高质量整合。
    """
    total_steps = 2
    flow_bar = tqdm(total=total_steps, desc="分析流程", unit="阶段", ncols=100) if tqdm else None
    if flow_bar:
        flow_bar.set_postfix_str("Phase1 采样与元协议")
    state, _, _ = run_phase1_sampling_and_protocol(
        book_id, title, chapters, total_chapter_count
    )
    if flow_bar:
        flow_bar.update(1)
        flow_bar.set_postfix_str("Phase2 逐章/窗口提取与整合")
    if use_per_chapter_extraction:
        if use_concurrent_extraction:
            state = run_phase2_concurrent_extract_then_consolidate(
                state, chapters, max_workers=max_concurrent_workers
            )
        else:
            state = run_phase2_per_chapter_then_consolidate(
                state, chapters, consolidate_every_n_chapters=consolidate_every_n_chapters
            )
    else:
        state = run_phase2_incremental_extract(
            state, chapters, refine_every_n_windows=refine_every_n_windows
        )
    if flow_bar:
        flow_bar.update(1)
        flow_bar.close()
    return state


def run_phase2_only(
    state: AnalysisState,
    chapters: List[dict],
    use_per_chapter_extraction: bool = True,
    use_concurrent_extraction: bool = True,
    max_concurrent_workers: int = 4,
    consolidate_every_n_chapters: int = 1,
    refine_every_n_windows: int = 1,
) -> AnalysisState:
    """
    仅执行第二阶段：沿用 state 中已有元协议，清空此前提取结果后重新逐章/窗口提取并整合。
    用于「只重跑章节提取、元协议不变」的场景。
    """
    state.cards = []
    state.plot_tree = {}
    state.conflict_marks = []
    state.last_processed_chapter_index = -1
    state.pending_patches = []
    state.template_version = 1
    state.last_refactor_at_chapter = -1
    if use_per_chapter_extraction:
        if use_concurrent_extraction:
            state = run_phase2_concurrent_extract_then_consolidate(
                state, chapters, max_workers=max_concurrent_workers
            )
        else:
            state = run_phase2_per_chapter_then_consolidate(
                state, chapters, consolidate_every_n_chapters=consolidate_every_n_chapters
            )
    else:
        state = run_phase2_incremental_extract(
            state, chapters, refine_every_n_windows=refine_every_n_windows
        )
    return state
