# -*- coding: utf-8 -*-
"""
LangGraph 人机在环工作流：Planner → Consultant → Writer → Critique，支持条件回退。
"""
from typing import Any, Dict, Literal, Optional

from .branch_simulation import generate_chapter_for_branch, get_three_plot_directions
from .state_schema import WriterState

try:
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
except ImportError:
    StateGraph = None
    START = "__start__"
    END = "__end__"


# 图状态：与 WriterState 字段对齐的 dict，便于节点返回局部更新
WriterGraphState = Dict[str, Any]


def _state_to_dict(s: WriterState) -> WriterGraphState:
    return s.model_dump() if hasattr(s, "model_dump") else s


def _dict_to_state(d: WriterGraphState) -> WriterState:
    return WriterState(**{k: d.get(k) for k in WriterState.model_fields if k in d})


def _user_probe_node(state: WriterGraphState) -> WriterGraphState:
    """节点 0：用户心理探测，将用户画像转为偏好补丁与走向提示，注入生成链路。"""
    try:
        from src.user_center import produce_user_context
        from src.user_center.schema import UserProfile
    except ImportError:
        return {}
    profile = state.get("_user_profile")
    if profile is None:
        return {}
    if isinstance(profile, dict):
        profile = UserProfile(**profile)
    ctx = produce_user_context(
        profile,
        book_id=state.get("book_id", ""),
        chapter_index=state.get("chapter_index", -1),
        chapter_type_hint=state.get("chapter_type", ""),
    )
    return {
        "preference_patch": ctx.preference_patch,
        "steering_hint": ctx.steering_hint or "",
    }


def _planner_node(state: WriterGraphState) -> WriterGraphState:
    """节点 1：提取用户意图，结合剧情树生成 3 个可选剧情走向。"""
    ws = _dict_to_state(state)
    directions = get_three_plot_directions(ws)
    return {"plot_directions": directions}


def _consultant_node(state: WriterGraphState) -> WriterGraphState:
    """节点 2：Librarian AI 对 3 个走向做因果合规性评分（简化：LLM 一次性打分）。"""
    from src.utils.llm_client import chat_high_quality

    directions = state.get("plot_directions") or []
    if not directions:
        return {}
    summaries = [d.get("summary", "") for d in directions]
    ctx = (state.get("history_causal_pack") or "")[:2000]
    user = f"""以下是一本书当前因果摘要与第 {state.get('chapter_index', 0) + 1} 章的三个剧情走向摘要。请对每个走向做因果合规性评分（0-10 整数），并只输出一行，格式：分数1,分数2,分数3。例如：8,6,9。

因果摘要：
{ctx}

走向1：{summaries[0]}
走向2：{summaries[1]}
走向3：{summaries[2]}
"""
    messages = [
        {"role": "system", "content": "你只输出一行三个分数，用英文逗号分隔。"},
        {"role": "user", "content": user},
    ]
    raw = (chat_high_quality(messages) or "").strip()
    scores = [None, None, None]
    if raw:
        parts = raw.replace("，", ",").split(",")
        for i, p in enumerate(parts[:3]):
            try:
                scores[i] = int(p.strip())
            except (ValueError, TypeError):
                pass
    updated = list(directions)
    for i, s in enumerate(scores):
        if i < len(updated) and s is not None:
            updated[i] = {**updated[i], "score": s}
    return {"plot_directions": updated}


def _writer_node(state: WriterGraphState) -> WriterGraphState:
    """节点 3：按选定走向执行战略→逻辑→草稿→润色，写入 draft。"""
    ws = _dict_to_state(state)
    analysis_state = state.get("_analysis_state")  # 由 invoke 方通过 config 注入
    style_injector = state.get("_style_injector")
    if analysis_state is None and ws.book_id:
        try:
            from src.analyzer.rewrite_pipeline import load_state_for_rewrite
            analysis_state = load_state_for_rewrite(ws.book_id)
        except Exception:
            analysis_state = None
    updated = generate_chapter_for_branch(
        ws,
        selected_branch_index=ws.selected_branch_index,
        analysis_state=analysis_state,
        style_injector=style_injector,
    )
    return {
        "beat_sheet": updated.beat_sheet,
        "chapter_type": updated.chapter_type,
        "logic_check_report": updated.logic_check_report,
        "constraint_boundaries": updated.constraint_boundaries,
        "draft": updated.draft,
        "style_samples": updated.style_samples,
        "polish_feedback": updated.polish_feedback,
    }


def _critique_node(state: WriterGraphState) -> WriterGraphState:
    """节点 4：批判智能体，检查续写是否崩人设/降智；不通过则回退 Writer 或 Planner。"""
    from src.utils.llm_client import chat_high_quality

    draft = (state.get("draft") or "")[:2500]
    rounds = state.get("_critique_rounds", 0) + 1
    if not draft:
        return {"critique_passed": True, "critique_feedback": "无草稿，跳过批判。", "_critique_rounds": rounds}
    user = f"""以下是一章网文续写草稿（前 2500 字）。请判断是否存在「崩人设」或「降智」问题（例如角色突然性格突变、做出明显不合理选择）。只输出一行 JSON：{{ "passed": true/false, "reason": "简短理由" }}。

草稿：
{draft}
"""
    messages = [
        {"role": "system", "content": "你是续写质量审核员，只输出一行 JSON。"},
        {"role": "user", "content": user},
    ]
    raw = (chat_high_quality(messages) or "").strip()
    for p in ("```json", "```"):
        if raw.startswith(p):
            raw = raw[len(p):].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    import json
    try:
        data = json.loads(raw)
        passed = data.get("passed", True)
        reason = data.get("reason", "")
    except Exception:
        passed = True
        reason = ""
    return {
        "critique_passed": passed,
        "critique_feedback": reason,
        "_critique_rounds": rounds,
    }


def _route_after_critique(state: WriterGraphState) -> Literal["writer", "planner", "__end__"]:
    """Critique 之后：通过则结束，否则根据反馈决定回退到 Writer 或 Planner；超过 max_rounds 强制结束。"""
    if state.get("critique_passed"):
        return "__end__"
    rounds = state.get("_critique_rounds", 0) + 1
    if rounds >= state.get("max_rounds", 3):
        return "__end__"
    feedback = (state.get("critique_feedback") or "").lower()
    if "大纲" in feedback or "走向" in feedback or "方向" in feedback:
        return "planner"
    return "writer"


def build_writer_graph(include_user_probe: bool = True):
    """构建（可选）用户心理探测 → Planner → Consultant → Writer → Critique 图。"""
    if StateGraph is None:
        raise ImportError("请安装 langgraph: pip install langgraph")
    graph = StateGraph(WriterGraphState)

    if include_user_probe:
        graph.add_node("user_probe", _user_probe_node)
        graph.add_edge(START, "user_probe")
        graph.add_edge("user_probe", "planner")
    else:
        graph.add_edge(START, "planner")

    graph.add_node("planner", _planner_node)
    graph.add_node("consultant", _consultant_node)
    graph.add_node("writer", _writer_node)
    graph.add_node("critique", _critique_node)

    graph.add_edge("planner", "consultant")
    # 用户选定分支后，从 consultant 到 writer（此处简化：默认 selected_branch_index 已在 state 中）
    graph.add_edge("consultant", "writer")
    graph.add_edge("writer", "critique")
    graph.add_conditional_edges("critique", _route_after_critique, {
        "writer": "writer",
        "planner": "planner",
        "__end__": END,
    })

    return graph.compile()


def run_writer_flow(
    state: WriterState,
    analysis_state: Optional[Any] = None,
    style_injector: Optional[Any] = None,
    max_rounds: int = 3,
) -> WriterState:
    """
    执行一次完整的人机在环写作流。入口为 WriterState（可含 user_intent、selected_branch_index 等）。
    若需传入已加载的 AnalysisState 或 StyleInjector，通过 config.configurable 注入。
    """
    graph = build_writer_graph()
    initial = _state_to_dict(state)
    initial["max_rounds"] = max_rounds
    # 通过 state 传递不可序列化对象，供 writer 节点使用
    if analysis_state is not None:
        initial["_analysis_state"] = analysis_state
    if style_injector is not None:
        initial["_style_injector"] = style_injector
    config = {"configurable": {}}
    final = graph.invoke(initial, config=config)
    return _dict_to_state({k: v for k, v in final.items() if not k.startswith("_")})
