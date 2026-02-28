# -*- coding: utf-8 -*-
"""
演化运行器 (Evolution Loop)：自动化「读原著 → 生成续写 → 评分 → 低分则诊断并等人改 → 每 10 章存档」。
"""
import json
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from .engineering import EngineeringDiagnostician
from .simulator import SimulatorAI, EvaluationResult
from .snapshot import save_snapshot, get_next_version
from .trace_logger import TraceLogger


# 默认阈值与步长
SCORE_THRESHOLD = 85
SNAPSHOT_EVERY_N_SUCCESS = 10


class EvolutionLoop:
    """
    主循环：读取原著第 N 章 → WriterAgent 生成续写 → SimulatorAI 评分 →
    若 score < 85 则触发 EngineeringDiagnostician 并暂停等人修改；
    若 score >= 85 则进入下一章；每连续成功 10 章调用 snapshot 存档。
    """

    def __init__(
        self,
        book_id: str,
        data_raw: Optional[Path] = None,
        data_cards: Optional[Path] = None,
        score_threshold: int = SCORE_THRESHOLD,
        snapshot_every_n: int = SNAPSHOT_EVERY_N_SUCCESS,
        on_pause: Optional[Callable[[str], None]] = None,
    ):
        root = Path(__file__).resolve().parents[2]
        self.book_id = book_id
        self.data_raw = Path(data_raw) if data_raw else root / "data" / "raw"
        self.data_cards = Path(data_cards) if data_cards else root / "data" / "cards"
        self.score_threshold = score_threshold
        self.snapshot_every_n = snapshot_every_n
        self.on_pause = on_pause  # 低分时调用，传入 action_plan，可打印并等待输入
        self.simulator = SimulatorAI(use_blind=True)
        self.diagnostician = EngineeringDiagnostician()
        self.trace_logger = TraceLogger()
        self._success_count = 0

    def _load_chapters(self) -> list:
        """加载书籍章节列表。"""
        p = self.data_raw / self.book_id / f"{self.book_id}.json"
        if not p.is_file():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("chapters") or []

    def _get_original_text(self, chapter_index: int) -> str:
        """获取原著第 chapter_index 章正文（0-based）。"""
        chapters = self._load_chapters()
        if chapter_index < 0 or chapter_index >= len(chapters):
            return ""
        ch = chapters[chapter_index]
        return (ch.get("content") or "").strip()

    def run_chapter(
        self,
        chapter_index: int,
        generate_fn: Callable[[str, int], tuple[str, str]],
    ) -> tuple[EvaluationResult, Optional[str], str]:
        """
        对给定章节执行：生成续写 → 评分。
        generate_fn(book_id, chapter_index) -> (generated_draft, run_id).
        返回 (evaluation_result, run_id, original_next_chapter_text)。
        """
        chapters = self._load_chapters()
        if chapter_index + 1 >= len(chapters):
            return (
                EvaluationResult(critique="无下一章原文可对比"),
                None,
                "",
            )
        original_next = self._get_original_text(chapter_index + 1)
        if not original_next:
            return (
                EvaluationResult(critique="下一章原文为空"),
                None,
                "",
            )
        run_id = str(uuid.uuid4())
        try:
            draft, run_id_out = generate_fn(self.book_id, chapter_index)
            run_id = run_id_out or run_id
        except Exception as e:
            self.trace_logger.log_event_sync(run_id, "runner", inputs={"chapter_index": chapter_index}, error=str(e))
            return (
                EvaluationResult(critique=f"生成异常: {e}"),
                run_id,
                original_next,
            )
        result = self.simulator.evaluate(original_text=original_next, generated_text=draft or "")
        return result, run_id, original_next

    def run(
        self,
        start_chapter: int = 0,
        max_chapters: Optional[int] = None,
        generate_fn: Optional[Callable[[str, int], tuple[str, str]]] = None,
    ) -> None:
        """
        主循环。generate_fn(book_id, chapter_index) 应返回 (generated_draft, run_id)。
        若未传入 generate_fn，则使用内置的 WriterAgent 调用（需可导入 writer 与 analyzer）。
        """
        if generate_fn is None:
            generate_fn = self._default_generate_fn
        chapters = self._load_chapters()
        if not chapters:
            raise FileNotFoundError(f"未找到书籍章节: {self.data_raw / self.book_id}")
        end = len(chapters) - 1
        if max_chapters is not None:
            end = min(end, start_chapter + max_chapters - 1)
        for chapter_index in range(start_chapter, end):
            result, run_id, _ = self.run_chapter(chapter_index, generate_fn)
            avg = SimulatorAI.average_score(result)
            if avg < self.score_threshold:
                trace_logs = self.trace_logger.get_trace(run_id) if run_id else []
                action_plan = self.diagnostician.diagnose(result, trace_logs)
                if self.on_pause:
                    self.on_pause(action_plan)
                else:
                    print("\n--- Action Plan (score < {}): ---\n{}\n--- 请修改代码后按 Enter 继续 ---".format(
                        self.score_threshold, action_plan
                    ))
                    input()
                continue
            self._success_count += 1
            if self._success_count >= self.snapshot_every_n:
                self._success_count = 0
                snap_path = save_snapshot(avg_score=avg, note=f"book={self.book_id} chapter={chapter_index}")
                print(f"已存档: {snap_path}")

    def _default_generate_fn(self, book_id: str, chapter_index: int) -> tuple[str, str]:
        """默认生成函数：调用 WriterAgent（需已有分析状态与三维上下文）。"""
        import uuid
        from src.analyzer import load_state_for_rewrite
        from src.librarian.context_loader import build_rewrite_context
        from src.librarian.style_store import StyleStore
        from src.writer import WriterState, generate_chapter_for_branch, get_three_plot_directions
        from src.writer.style_injector import StyleInjector

        run_id = str(uuid.uuid4())
        analysis_state = load_state_for_rewrite(book_id, self.data_cards)
        if not analysis_state:
            return "", run_id
        ctx = build_rewrite_context(
            analysis_state,
            rewritten_chapter_index=chapter_index,
            new_anchors_text="续写下一章",
        )
        state = WriterState(
            book_id=book_id,
            chapter_index=chapter_index,
            user_intent="续写下一章",
            history_causal_pack=ctx["history_causal_pack"],
            new_causal_anchors=ctx["new_causal_anchors"],
            rule_constraints_pack=ctx["rule_constraints_pack"],
            selected_branch_index=0,
        )
        style_store = StyleStore(book_id=book_id)
        book_json = self.data_raw / book_id / f"{book_id}.json"
        fingerprint_file = self.data_cards / book_id / "style_fingerprint.json"
        if book_json.is_file():
            style_store.load_from_book_json(book_json, fingerprint_file=fingerprint_file)
        injector = StyleInjector(style_store=style_store)
        state.plot_directions = get_three_plot_directions(state)
        state = generate_chapter_for_branch(state, 0, analysis_state, injector)
        self.trace_logger.log_event_sync(
            run_id,
            "writer",
            inputs={"book_id": book_id, "chapter_index": chapter_index},
            outputs={"draft_length": len(state.draft or "")},
            context_snapshot=[s.get("preview", "")[:100] for s in (state.style_samples or [])],
        )
        return state.draft or "", run_id
