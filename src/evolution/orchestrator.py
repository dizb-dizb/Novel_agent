# -*- coding: utf-8 -*-
"""
主控 (Orchestrator)：自动驾驶环。循环执行「运行评测 → 不合格则诊断并改代码 → 合格则快照」直到达标或达最大迭代。
"""
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.utils.llm_client import chat_high_quality

from .engineering import EngineeringDiagnostician
from .file_operator import (
    PROJECT_ROOT,
    apply_edits,
    extract_paths_from_markdown,
    resolve_safe,
)
from .runner import EvolutionLoop
from .simulator import EvaluationResult, SimulatorAI
from .snapshot import save_snapshot


def plan_to_edits(
    action_plan_md: str,
    project_root: Path,
    max_files_per_round: int = 2,
) -> List[Dict[str, Any]]:
    """
    根据 Action Plan 的 Markdown 与当前文件内容，调用 LLM 生成可应用的编辑列表。
    每个编辑为 {"path": "src/...", "new_content": "完整文件内容"}。
    """
    paths = extract_paths_from_markdown(action_plan_md)
    if not paths:
        return []
    root = Path(project_root)
    edits = []
    for path in paths[:max_files_per_round]:
        target = resolve_safe(root, path)
        if not target or not target.is_file():
            continue
        try:
            current = target.read_text(encoding="utf-8")
        except Exception:
            continue
        user = f"""你收到一份代码修改建议（Action Plan）。请根据建议，输出**唯一**需要修改的文件之**完整新内容**。

## Action Plan（节选）
{action_plan_md[:4000]}

## 当前文件路径与内容
路径：{path}

```
{current[:6000]}
```

要求：
1. 只输出该文件的完整新代码，不要用 ```python 包裹，不要解释。
2. 若无需修改此文件，请输出 exactly: NO_CHANGE
"""
        messages = [
            {"role": "system", "content": "你只输出修改后的完整文件内容，或 NO_CHANGE。"},
            {"role": "user", "content": user},
        ]
        raw = (chat_high_quality(messages) or "").strip()
        if "NO_CHANGE" in raw and len(raw) < 20:
            continue
        for wrap in ("```python", "```"):
            if raw.startswith(wrap):
                raw = raw[len(wrap):].strip()
            if raw.endswith("```"):
                raw = raw[:-3].strip()
        if len(raw) > 100:
            edits.append({"path": path, "new_content": raw})
    return edits


def run_evolution_loop(
    book_id: str,
    chapter_index: int = 0,
    score_threshold: float = 85.0,
    max_iterations: int = 10,
    data_raw: Optional[Path] = None,
    data_cards: Optional[Path] = None,
    generate_fn: Optional[Callable[[str, int], Tuple[str, str]]] = None,
    on_iteration: Optional[Callable[[int, float, Optional[str]], None]] = None,
) -> Tuple[bool, float, int]:
    """
    执行一轮自动驾驶优化环：
    1. 运行 runner 对指定章节做续写测试并收集评分。
    2. 若评分不合格，将评分与 trace 交给工程诊断 AI，得到 Action Plan。
    3. 将 Action Plan 转为 edits，调用 file_operator 修改源码。
    4. 回到步骤 1，直到评分达标或达到 max_iterations。
    5. 每次评分达标时调用 save_snapshot 备份。

    返回：(是否达标, 最终平均分, 迭代次数)。
    """
    root = PROJECT_ROOT
    loop = EvolutionLoop(
        book_id=book_id,
        data_raw=data_raw,
        data_cards=data_cards,
        score_threshold=int(score_threshold),
        snapshot_every_n=1,
        on_pause=None,
    )
    if generate_fn is None:
        generate_fn = loop._default_generate_fn
    diagnostician = EngineeringDiagnostician()
    best_score = 0.0
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        result, run_id, _ = loop.run_chapter(chapter_index, generate_fn)
        avg = SimulatorAI.average_score(result)
        best_score = max(best_score, avg)
        if on_iteration:
            on_iteration(iteration, avg, result.critique)
        if avg >= score_threshold:
            save_snapshot(version=None, avg_score=avg, note=f"book={book_id} ch={chapter_index} iter={iteration}")
            return True, avg, iteration
        trace_logs = loop.trace_logger.get_trace(run_id) if run_id else []
        action_plan = diagnostician.diagnose(result, trace_logs)
        if on_iteration:
            on_iteration(iteration, avg, action_plan[:500])
        edits = plan_to_edits(action_plan, root, max_files_per_round=2)
        if not edits:
            if on_iteration:
                on_iteration(iteration, avg, "无可行编辑，退出")
            break
        outcomes = apply_edits(root, edits)
        if not any(ok for ok, _ in outcomes):
            break
    return False, best_score, iteration


def main_cli() -> None:
    """供 subprocess 或命令行调用。"""
    import argparse
    p = argparse.ArgumentParser(description="Evolution Orchestrator: 自动驾驶优化环")
    p.add_argument("--book-id", required=True, help="书籍 id")
    p.add_argument("--chapter", type=int, default=0, help="章节索引 0-based")
    p.add_argument("--threshold", type=float, default=85.0, help="达标分数")
    p.add_argument("--max-iter", type=int, default=10, help="最大迭代次数")
    args = p.parse_args()

    def on_it(n: int, score: float, msg: Optional[str]) -> None:
        print(f"[iter {n}] score={score:.1f} {msg[:80] if msg else ''}")

    ok, score, it = run_evolution_loop(
        book_id=args.book_id,
        chapter_index=args.chapter,
        score_threshold=args.threshold,
        max_iterations=args.max_iter,
        on_iteration=on_it,
    )
    print(f"Done: passed={ok}, best_score={score:.1f}, iterations={it}")


if __name__ == "__main__":
    main_cli()
