# -*- coding: utf-8 -*-
"""
LangGraph 检测模块：以 stream 方式运行写作图，逐节点打印状态与输出，便于调试与观察。
"""
from typing import Any, Dict, Iterator, Optional, Tuple

from .graph_workflow import (
    WriterGraphState,
    _dict_to_state,
    _state_to_dict,
    build_writer_graph,
)
from .state_schema import WriterState


# 用于打印的 state 字段摘要（避免整段 draft 刷屏）
SUMMARY_KEYS = (
    "book_id",
    "chapter_index",
    "user_intent",
    "preference_patch",
    "steering_hint",
    "chapter_type",
    "logic_check_report",
    "constraint_boundaries",
    "critique_passed",
    "critique_feedback",
    "polish_feedback",
    "max_rounds",
    "_critique_rounds",
)


def _summary(state: Dict[str, Any], max_str_len: int = 200) -> Dict[str, Any]:
    """生成可读的状态摘要，长字符串截断。"""
    out = {}
    for k in SUMMARY_KEYS:
        if k not in state:
            continue
        v = state[k]
        if isinstance(v, str) and len(v) > max_str_len:
            v = v[:max_str_len] + "…"
        elif isinstance(v, list) and k == "plot_directions":
            v = [
                {"summary": (d.get("summary") or "")[:80], "score": d.get("score")}
                for d in (v or [])[:3]
            ]
        out[k] = v
    if "draft" in state and state["draft"]:
        out["draft_len"] = len(state["draft"])
        out["draft_preview"] = (state["draft"] or "")[:150] + "…"
    if "beat_sheet" in state and state["beat_sheet"]:
        out["beat_sheet_preview"] = (state["beat_sheet"] or "")[:150] + "…"
    return out


def stream_writer_graph(
    initial_state: WriterState,
    analysis_state: Optional[Any] = None,
    style_injector: Optional[Any] = None,
    max_rounds: int = 3,
    stream_mode: str = "updates",
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    以 stream 方式运行写作图，逐节点 yield (node_name, state_or_update)。
    stream_mode: "updates" 仅该节点产出、 "values" 为当前全量状态。
    返回生成器：(节点名, 状态或更新字典)。
    """
    graph = build_writer_graph()
    initial = _state_to_dict(initial_state)
    initial["max_rounds"] = max_rounds
    if analysis_state is not None:
        initial["_analysis_state"] = analysis_state
    if style_injector is not None:
        initial["_style_injector"] = style_injector
    config = {"configurable": {}}

    try:
        stream = graph.stream(initial, config=config, stream_mode=stream_mode)
    except TypeError:
        # 旧版 langgraph 可能无 stream_mode
        stream = graph.stream(initial, config=config)

    full_state = dict(initial)
    for chunk in stream:
        if not isinstance(chunk, dict):
            yield ("?", chunk)
            continue
        # chunk 可能是 { "planner": {...}, "consultant": {...} } 或 全量 state
        for node_name, value in chunk.items():
            if isinstance(value, dict):
                full_state.update(value)
            if stream_mode == "values":
                yield (node_name, dict(full_state))
            else:
                yield (node_name, value if isinstance(value, dict) else {node_name: value})
    return


def run_with_inspection(
    state: WriterState,
    analysis_state: Optional[Any] = None,
    style_injector: Optional[Any] = None,
    max_rounds: int = 3,
    verbose: bool = True,
) -> WriterState:
    """
    运行写作图并在每步打印节点名与状态摘要；最后返回与 run_writer_flow 相同的 WriterState。
    verbose=True 时在控制台输出每节点摘要。
    """
    graph = build_writer_graph()
    initial = _state_to_dict(state)
    initial["max_rounds"] = max_rounds
    if analysis_state is not None:
        initial["_analysis_state"] = analysis_state
    if style_injector is not None:
        initial["_style_injector"] = style_injector
    config = {"configurable": {}}

    full_state = dict(initial)
    try:
        stream = graph.stream(initial, config=config, stream_mode="updates")
    except TypeError:
        stream = graph.stream(initial, config=config)

    step = 0
    for chunk in stream:
        if not isinstance(chunk, dict):
            if verbose:
                print(f"[Step {step}] chunk type: {type(chunk)}")
            step += 1
            continue
        for node_name, value in chunk.items():
            if isinstance(value, dict):
                full_state.update(value)
            if verbose:
                summary = _summary(full_state)
                print(f"\n--- 节点: {node_name} ---")
                for k, v in summary.items():
                    if k.startswith("_"):
                        continue
                    print(f"  {k}: {v}")
            step += 1

    clean = {k: v for k, v in full_state.items() if not k.startswith("_")}
    return _dict_to_state(clean)


def main_cli():
    """命令行：用指定 book_id / chapter 跑一遍带检测的写作流（仅 preview 或单步）。"""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="LangGraph 写作图检测：逐节点输出状态")
    parser.add_argument("--book-id", required=True, help="书籍 id")
    parser.add_argument("--chapter", type=int, default=0, help="章节索引 0-based")
    parser.add_argument("--intent", default="续写下一章", help="用户意图")
    parser.add_argument("--branch", type=int, default=0, help="选定走向 0/1/2")
    parser.add_argument("--quiet", action="store_true", help="不打印每步摘要，仅最后结果")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_cards = root / "data" / "cards"
    data_raw = root / "data" / "raw"

    from src.analyzer.rewrite_pipeline import load_state_for_rewrite
    from src.librarian.context_loader import build_rewrite_context
    from src.librarian.style_store import StyleStore
    from src.writer.style_injector import StyleInjector

    analysis_state = load_state_for_rewrite(args.book_id, data_cards)
    if not analysis_state:
        print("未找到 analysis_state，请先运行: python main.py analyze --book-id", args.book_id)
        return 1
    ctx = build_rewrite_context(
        analysis_state,
        rewritten_chapter_index=args.chapter,
        new_anchors_text=args.intent,
    )
    state = WriterState(
        book_id=args.book_id,
        chapter_index=args.chapter,
        user_intent=args.intent,
        history_causal_pack=ctx["history_causal_pack"],
        new_causal_anchors=ctx["new_causal_anchors"],
        rule_constraints_pack=ctx["rule_constraints_pack"],
        selected_branch_index=args.branch,
    )
    style_store = StyleStore(book_id=args.book_id)
    book_json = data_raw / args.book_id / f"{args.book_id}.json"
    fingerprint_file = data_cards / args.book_id / "style_fingerprint.json"
    if book_json.is_file():
        style_store.load_from_book_json(book_json, fingerprint_file=fingerprint_file)
    style_injector = StyleInjector(style_store=style_store)

    final = run_with_inspection(
        state,
        analysis_state=analysis_state,
        style_injector=style_injector,
        max_rounds=3,
        verbose=not args.quiet,
    )
    print("\n========== 最终结果 ==========")
    print("chapter_type:", final.chapter_type)
    print("critique_passed:", final.critique_passed)
    print("draft 长度:", len(final.draft or ""))
    if final.draft:
        print("draft 前 300 字:", (final.draft or "")[:300])
    return 0


if __name__ == "__main__":
    exit(main_cli())
