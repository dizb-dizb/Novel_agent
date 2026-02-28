# -*- coding: utf-8 -*-
"""
完整基准测试：使用导入的参考书，执行「分析 → 仿写」全流程并记录耗时与结果。
默认使用 data/raw/7512268698271370302 下的导入书籍。

运行（项目根目录）：
  python -m backend.engine.run_benchmark
  python -m backend.engine.run_benchmark --book-id 7512268698271370302 --analyze-chapters 15 --imitation-chapters 3
  python -m backend.engine.run_benchmark --skip-analyze --imitation-chapters 5   # 仅仿写，沿用已有知识库
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")

# 默认使用你导入的那本书
DEFAULT_BOOK_ID = "7512268698271370302"


async def run_full_benchmark(
    book_id: str,
    *,
    analyze_chapters: int = 15,
    imitation_chapters: int = 3,
    skip_analyze: bool = False,
    force_analyze: bool = False,
    output_report: str = "",
) -> dict:
    """
    全流程基准测试：Phase 0 分析（可选/强制）→ Phase 1 仿写 N 章。
    返回统计 dict：phase0_sec, phase1_sec, phase1_per_chapter_sec, chapters_done, success, error.
    """
    from backend.engine.auto_novel_engine import (
        AutoNovelEngine,
        DefaultAnalysisAgent,
        assemble_rag_from_backend,
    )
    from backend.engine.imitation_pipeline import (
        ensure_knowledge_bases_and_style,
        load_style_guide_from_fingerprint_file,
    )
    from backend.engine.logic_master import LogicMasterAgent
    from backend.schemas.orchestrator_models import BookState, StyleGuide
    from backend.services.writing_service import WritingAgent

    raw_dir = _ROOT / "data" / "raw"
    cards_dir = _ROOT / "data" / "cards"
    json_path = raw_dir / book_id / f"{book_id}.json"
    novel_db_path = cards_dir / book_id / "novel_database.json"

    report = {
        "book_id": book_id,
        "phase0_sec": 0.0,
        "phase1_sec": 0.0,
        "phase1_per_chapter_sec": [],
        "chapters_done": 0,
        "success": False,
        "error": None,
    }

    # ---------- 环境检查 ----------
    if not (os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")):
        report["error"] = "未设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY"
        return report
    if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")):
        logger.warning("未设置 DASHSCOPE_API_KEY，文笔阶段可能失败")

    # ---------- Phase 0：分析（形成四大知识库 + 文笔指纹） ----------
    style_guide: StyleGuide
    run_phase0 = not skip_analyze and (force_analyze or not novel_db_path.is_file())
    if run_phase0:
        if not json_path.is_file():
            report["error"] = f"未找到参考书原文: {json_path}，无法执行分析"
            return report
        logger.info("======== 基准测试 Phase 0：分析（前 %s 章）========", analyze_chapters)
        t0 = time.perf_counter()
        try:
            style_guide, _ = await ensure_knowledge_bases_and_style(
                book_id,
                raw_dir=raw_dir,
                cards_dir=cards_dir,
                force_analyze=force_analyze,
                write_db=True,
                max_chapters=analyze_chapters,
            )
            report["phase0_sec"] = round(time.perf_counter() - t0, 2)
            logger.info("Phase 0 完成，耗时 %.1f 秒", report["phase0_sec"])
        except Exception as e:
            report["phase0_sec"] = round(time.perf_counter() - t0, 2)
            report["error"] = f"Phase 0 失败: {e}"
            logger.exception("%s", report["error"])
            return report
    elif novel_db_path.is_file():
        fp_path = cards_dir / book_id / "style_fingerprint.json"
        style_guide = load_style_guide_from_fingerprint_file(fp_path)
        if not style_guide:
            style_guide = StyleGuide(
                reference_book_name=book_id,
                vocabulary_features=[],
                pacing_rules="",
                dialogue_style="",
            )
        logger.info("跳过 Phase 0，使用已有知识库: %s", novel_db_path)
    else:
        report["error"] = f"未检测到知识库且未提供原文。请先运行分析或放置 {json_path}"
        return report

    # ---------- Phase 1：仿写 N 章 ----------
    if imitation_chapters < 1:
        report["success"] = True
        _write_report(report, output_report)
        return report

    logger.info("======== 基准测试 Phase 1：仿写 %s 章 ========", imitation_chapters)
    logic_master = LogicMasterAgent()
    writing_agent = WritingAgent(context_assembler=assemble_rag_from_backend)
    analysis_agent = DefaultAnalysisAgent()
    engine = AutoNovelEngine(
        logic_master=logic_master,
        writing_agent=writing_agent,
        analysis_agent=analysis_agent,
    )
    state = BookState(current_chapter=0, main_plot_goal="")
    t1 = time.perf_counter()
    try:
        final_state = await engine.run_imitation_loop(
            book_id,
            imitation_chapters,
            style_guide=style_guide,
            initial_state=state,
        )
        report["phase1_sec"] = round(time.perf_counter() - t1, 2)
        report["chapters_done"] = final_state.current_chapter
        report["success"] = True
        if report["chapters_done"]:
            report["phase1_per_chapter_sec"] = [round(report["phase1_sec"] / report["chapters_done"], 2)]
        logger.info("Phase 1 完成，总耗时 %.1f 秒，共 %s 章", report["phase1_sec"], report["chapters_done"])
    except Exception as e:
        report["phase1_sec"] = round(time.perf_counter() - t1, 2)
        report["error"] = f"Phase 1 失败: {e}"
        logger.exception("%s", report["error"])

    _write_report(report, output_report)
    return report


def _write_report(report: dict, path: str) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("基准报告已写入: %s", p)


def main() -> None:
    parser = argparse.ArgumentParser(description="完整基准测试：分析 → 仿写（使用导入的参考书）")
    parser.add_argument("--book-id", default=DEFAULT_BOOK_ID, help="书籍 ID（默认你导入的书）")
    parser.add_argument("--analyze-chapters", type=int, default=15, help="分析阶段使用的章节数（默认 15）")
    parser.add_argument("--imitation-chapters", type=int, default=3, help="仿写阶段生成的章节数（默认 3）")
    parser.add_argument("--skip-analyze", action="store_true", help="跳过分析，仅仿写（需已有知识库）")
    parser.add_argument("--force-analyze", action="store_true", help="强制先重新分析再仿写")
    parser.add_argument("--output-report", default="", help="将报告写入该文件，如 data/benchmark_report.json")
    args = parser.parse_args()

    report = asyncio.run(
        run_full_benchmark(
            args.book_id,
            analyze_chapters=args.analyze_chapters,
            imitation_chapters=args.imitation_chapters,
            skip_analyze=args.skip_analyze,
            force_analyze=args.force_analyze,
            output_report=args.output_report or str(_ROOT / "data" / "benchmark_report.json"),
        )
    )

    print("\n======== 基准测试结果 ========")
    print("书籍 ID:", report["book_id"])
    print("Phase 0 耗时(秒):", report["phase0_sec"])
    print("Phase 1 耗时(秒):", report["phase1_sec"])
    print("仿写完成章数:", report["chapters_done"])
    if report.get("phase1_per_chapter_sec"):
        print("每章耗时(秒):", report["phase1_per_chapter_sec"])
    print("成功:", report["success"])
    if report.get("error"):
        print("错误:", report["error"])
    print("报告文件:", args.output_report or "data/benchmark_report.json")
    sys.exit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
