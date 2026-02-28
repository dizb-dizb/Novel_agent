# -*- coding: utf-8 -*-
"""
仿写整本书流程：必须先运行分析模块形成四大知识库 + 文笔指纹，再基于知识库与文笔进行仿写。
未运行分析时没有知识库，脚本会报错退出，不会用占位数据仿写。

运行方式（在项目根目录）：
  # 方式一：先有参考书原文 data/raw/{book_id}/{book_id}.json，本脚本会自动先跑分析再仿写
  python -m backend.engine.run_imitation_test --book-id 7512268698271370302 --chapters 1
  # 方式二：先单独跑分析（main.py analyze），再跑本脚本（会读取已有知识库）
  python main.py analyze --book-id 7512268698271370302 --chapters 10 --write-db
  python -m backend.engine.run_imitation_test --book-id 7512268698271370302 --chapters 1
  # 强制重新分析再仿写
  python -m backend.engine.run_imitation_test --book-id 7512268698271370302 --chapters 1 --force-analyze

环境变量：DEEPSEEK_API_KEY（逻辑审查）、DASHSCOPE_API_KEY（文笔），分析阶段会用到 ANALYZER_*。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

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
logger = logging.getLogger("imitation_test")


def get_book_total_chapters(book_id: str, raw_dir: Path, cards_dir: Path) -> int:
    """从原始书籍 JSON 或分析状态获取全书总章数；无数据时返回 0。"""
    json_path = raw_dir / book_id / f"{book_id}.json"
    if json_path.is_file():
        try:
            import json
            data = json.loads(json_path.read_text(encoding="utf-8"))
            return len(data.get("chapters") or [])
        except Exception:
            pass
    state_path = cards_dir / book_id / "analysis_state.json"
    if state_path.is_file():
        try:
            import json
            data = json.loads(state_path.read_text(encoding="utf-8"))
            cards = data.get("cards") or []
            if cards:
                indices = [c.get("chapter_index", 0) for c in cards if isinstance(c, dict)]
                return max(indices, default=0) + 1
        except Exception:
            pass
    return 0


async def run_imitation_flow(
    book_id: str,
    target_chapters: int,
    *,
    force_analyze: bool = False,
    no_seed: bool = False,
    analyze_only: bool = False,
) -> None:
    """
    完整仿写流程：
    Phase 0：若未建库或 --force-analyze，则先执行分析 → 四大知识库写入后端 + 文笔指纹。
    Phase 1：用知识库快照做逻辑审查、用知识库 RAG + 文笔指纹做受控编写，每章后分析固化。
    """
    from backend.engine.auto_novel_engine import (
        AutoNovelEngine,
        DefaultAnalysisAgent,
        assemble_rag_from_backend,
    )
    from backend.engine.imitation_pipeline import ensure_knowledge_bases_and_style
    from backend.engine.logic_master import LogicMasterAgent
    from backend.schemas.orchestrator_models import BookState, StyleGuide
    from backend.services.writing_service import WritingAgent

    if not (os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")):
        logger.error("请设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY（逻辑审查）")
        sys.exit(1)
    if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")):
        logger.warning("未设置 DASHSCOPE_API_KEY，文笔阶段可能失败（Qwen-max）")

    raw_dir = _ROOT / "data" / "raw"
    cards_dir = _ROOT / "data" / "cards"
    json_path = raw_dir / book_id / f"{book_id}.json"
    novel_db_path = cards_dir / book_id / "novel_database.json"

    # ---------- Phase 0：先分析形成四大知识库 + 文笔指纹 ----------
    style_guide: StyleGuide
    if json_path.is_file() and (force_analyze or not novel_db_path.is_file()):
        logger.info("Phase 0: 执行分析流水线 → 四大知识库 + 文笔指纹")
        try:
            style_guide, _ = await ensure_knowledge_bases_and_style(
                book_id,
                raw_dir=raw_dir,
                cards_dir=cards_dir,
                force_analyze=force_analyze,
                write_db=True,
            )
        except FileNotFoundError as e:
            logger.error("%s", e)
            sys.exit(1)
        except Exception as e:
            logger.exception("Phase 0 分析失败: %s", e)
            raise
        logger.info("Phase 0 完成: 知识库与文笔指纹已就绪，参考书名=%s", style_guide.reference_book_name)
    elif novel_db_path.is_file():
        from backend.engine.imitation_pipeline import load_style_guide_from_fingerprint_file
        fp_path = cards_dir / book_id / "style_fingerprint.json"
        style_guide = load_style_guide_from_fingerprint_file(fp_path)
        if not style_guide:
            style_guide = StyleGuide(
                reference_book_name=book_id,
                vocabulary_features=[],
                pacing_rules="",
                dialogue_style="",
            )
        logger.info(
            "Phase 0: 使用已有知识库（来自 %s），若需重新分析请加 --force-analyze",
            novel_db_path,
        )
    else:
        # 未运行过分析 → 没有知识库 → 不允许仿写
        logger.error(
            "未检测到知识库（分析模块尚未对该书运行）。\n"
            "仿写必须基于分析产出的四大知识库与文笔指纹，请先完成其一：\n"
            "  1) 放置参考书原文: %s\n"
            "     然后重新运行本脚本（将自动执行分析再仿写）；\n"
            "  2) 或先执行: python main.py analyze --book-id %s --write-db\n"
            "     再运行本脚本。",
            json_path,
            book_id,
        )
        sys.exit(1)

    if analyze_only:
        logger.info("--analyze-only: 仅执行分析，已结束")
        return

    # ---------- Phase 1：仿写循环（逻辑审查 + 知识库 RAG + 文笔指纹 + 分析固化） ----------
    logic_master = LogicMasterAgent()
    writing_agent = WritingAgent(context_assembler=assemble_rag_from_backend)
    analysis_agent = DefaultAnalysisAgent()
    engine = AutoNovelEngine(
        logic_master=logic_master,
        writing_agent=writing_agent,
        analysis_agent=analysis_agent,
    )

    initial_state = BookState(current_chapter=0, main_plot_goal="")
    logger.info("开始仿写流水线: book_id=%s, target_chapters=%s（知识库 RAG + 文笔指纹）", book_id, target_chapters)
    try:
        final_state = await engine.run_imitation_loop(
            book_id,
            target_chapters,
            style_guide=style_guide,
            initial_state=initial_state,
        )
        logger.info("流水线完成: 当前进度=%s, main_plot_goal=%s", final_state.current_chapter, final_state.main_plot_goal)
    except Exception as e:
        logger.exception("流水线异常: %s", e)
        raise


async def run_reconstruction_flow(
    book_id: str,
    *,
    max_chapters: Optional[int] = None,
    force_analyze: bool = False,
    use_chapter_word_count: bool = True,
    skip_framework_build: bool = False,
    outline_path: Optional[Path] = None,
    use_concurrent: bool = False,
    render_workers: int = 3,
) -> None:
    """
    逆向重构流程（两阶段）：
    Phase 1 - 整体搭建知识框架：全书 N 章在知识层面一次性完成逻辑适配，产出 N 个细纲并落盘。
    Phase 2 - 文本实现：编写模块仅按已落盘框架逐章渲染 + 分析固化，不再做逻辑设计。
    500 章即仿建 500 章的内容逻辑再写 500 章正文。
    """
    from backend.engine.auto_novel_engine import (
        AutoNovelEngine,
        DefaultAnalysisAgent,
        assemble_rag_from_backend,
    )
    from backend.engine.framework_builder import build_reconstructed_framework
    from backend.engine.imitation_pipeline import ensure_knowledge_bases_and_style
    from backend.engine.logic_master import LogicMasterAgent
    from backend.engine.plot_tree_loader import (
        build_original_plot_tree,
        get_chapter_word_counts_from_raw_book,
    )
    from backend.schemas.orchestrator_models import MutationPremise, StyleGuide
    from backend.services.writing_service import WritingAgent

    if not (os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")):
        logger.error("请设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY（逻辑审查）")
        sys.exit(1)
    if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")):
        logger.warning("未设置 DASHSCOPE_API_KEY，文笔阶段可能失败（Qwen-max）")

    raw_dir = _ROOT / "data" / "raw"
    cards_dir = _ROOT / "data" / "cards"
    novel_db_path = cards_dir / book_id / "novel_database.json"
    json_path = raw_dir / book_id / f"{book_id}.json"

    # ---------- Phase 0：确保四大知识库 + 文笔指纹 ----------
    if json_path.is_file() and (force_analyze or not novel_db_path.is_file()):
        logger.info("Phase 0: 执行分析流水线 → 四大知识库 + 文笔指纹")
        style_guide, _ = await ensure_knowledge_bases_and_style(
            book_id, raw_dir=raw_dir, cards_dir=cards_dir,
            force_analyze=force_analyze, write_db=True,
        )
    elif novel_db_path.is_file():
        from backend.engine.imitation_pipeline import load_style_guide_from_fingerprint_file
        fp_path = cards_dir / book_id / "style_fingerprint.json"
        style_guide = load_style_guide_from_fingerprint_file(fp_path)
        if not style_guide:
            style_guide = StyleGuide(
                reference_book_name=book_id,
                vocabulary_features=[],
                pacing_rules="",
                dialogue_style="",
                avg_chapter_length=None,
            )
        logger.info("Phase 0: 使用已有知识库与文笔指纹: %s", novel_db_path)
    else:
        logger.error("未检测到知识库，请先运行: python main.py analyze --book-id %s --write-db", book_id)
        sys.exit(1)

    logic_master = LogicMasterAgent()
    writing_agent = WritingAgent(context_assembler=assemble_rag_from_backend)
    analysis_agent = DefaultAnalysisAgent()
    engine = AutoNovelEngine(
        logic_master=logic_master,
        writing_agent=writing_agent,
        analysis_agent=analysis_agent,
    )

    outline_file = outline_path or (cards_dir / book_id / "reconstructed_outline.json")

    # ---------- Phase 1：整体搭建知识框架（全书 N 章一次性适配，落盘） ----------
    if not skip_framework_build:
        # 优先用 novel_database 的 plot_tree；若因整合只有少量关键章（如 9 章），则用原始书按章构建 N 章
        original_plot_tree = build_original_plot_tree(
            novel_db_path,
            raw_book_path=json_path if json_path.is_file() else None,
            max_chapters=max_chapters,
            prefer_full_chapter_list=True,
        )
        if not original_plot_tree:
            logger.error("无法构建原著情节树（novel_database 与原始书籍均不可用或为空）")
            sys.exit(1)
        logger.info("已构建原著情节树: 共 %s 章（整体搭建知识框架）", len(original_plot_tree))

        mutation_premise = MutationPremise(
            new_world_setting="保持原著世界观与设定，仅做同构仿写与文笔复现。",
            character_mapping={},
            core_rule_changes=[],
        )
        logger.info("Phase 1: 整体搭建知识框架（%s 章逻辑适配 → 落盘）", len(original_plot_tree))
        await build_reconstructed_framework(
            book_id,
            original_plot_tree,
            mutation_premise,
            logic_master,
            engine.get_db_snapshot,
            outline_path=outline_file,
            cards_dir=cards_dir,
        )
    elif not outline_file.is_file():
        logger.error("未跳过框架构建但未找到已落盘框架: %s；请先运行一次不传 --skip-framework", outline_file)
        sys.exit(1)

    # ---------- Phase 2：按框架逐章文本实现（仅编写 + 固化），可选并发 ----------
    logger.info(
        "Phase 2: 按知识框架逐章文本实现（%s）",
        f"{render_workers} 个打字机并行 + 后台入库" if use_concurrent else "逐章串行",
    )
    try:
        if use_concurrent:
            final_state = await engine.run_render_from_outline_concurrent(
                book_id,
                outline_file,
                style_guide=style_guide,
                render_workers=render_workers,
            )
        else:
            final_state = await engine.run_render_from_outline(
                book_id,
                outline_file,
                style_guide=style_guide,
            )
        logger.info("逆向重构流水线完成: 知识框架 %s 章，已实现文本 %s 章", final_state.current_chapter, final_state.current_chapter)
    except Exception as e:
        logger.exception("逆向重构流水线异常: %s", e)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="仿写整本书：先分析形成四大知识库与文笔指纹，再结合知识库与文笔完成仿写"
    )
    parser.add_argument("--book-id", default="7512268698271370302", help="书籍 ID")
    parser.add_argument("--chapters", type=int, default=None, help="目标章节数（未指定且非 --full-book 时默认 50）")
    parser.add_argument("--full-book", action="store_true", help="仿写整本书：目标章数从原始书籍或分析状态自动推断")
    parser.add_argument("--force-analyze", action="store_true", help="强制先执行分析再仿写（覆盖已有知识库）")
    parser.add_argument("--no-seed", action="store_true", help="不写入种子角色（仅当无分析结果且需兼容时使用）")
    parser.add_argument("--analyze-only", action="store_true", help="仅执行分析建库与文笔指纹，不跑仿写循环")
    parser.add_argument("--reconstruction", action="store_true", help="逆向重构：先整体搭建 N 章知识框架并落盘，再按框架逐章文本实现")
    parser.add_argument("--reconstruction-chapters", type=int, default=None, help="逆向重构时最多章节数（默认全书）；测试可设 2")
    parser.add_argument("--skip-framework", action="store_true", help="仅运行文本实现阶段，使用已落盘的 reconstructed_outline.json")
    parser.add_argument("--concurrent", action="store_true", help="Phase 2 使用并发流水线（多打字机并行渲染 + 后台入库）")
    parser.add_argument("--render-workers", type=int, default=3, help="并发渲染 worker 数（默认 3），仅 --concurrent 时有效")
    args = parser.parse_args()

    raw_dir = _ROOT / "data" / "raw"
    cards_dir = _ROOT / "data" / "cards"
    if args.reconstruction:
        asyncio.run(
            run_reconstruction_flow(
                args.book_id,
                max_chapters=args.reconstruction_chapters,
                force_analyze=args.force_analyze,
                use_chapter_word_count=True,
                skip_framework_build=getattr(args, "skip_framework", False),
                use_concurrent=getattr(args, "concurrent", False),
                render_workers=getattr(args, "render_workers", 3),
            )
        )
        return
    if args.full_book:
        target_chapters = get_book_total_chapters(args.book_id, raw_dir, cards_dir)
        if target_chapters < 1:
            logger.error("--full-book 需要已有书籍原文或分析状态以获取总章数。请确保 data/raw/%s 或 data/cards/%s 存在。", args.book_id, args.book_id)
            sys.exit(1)
        logger.info("整本书仿写: 目标 %s 章（从书籍/分析状态推断）", target_chapters)
    else:
        target_chapters = args.chapters if args.chapters is not None else 50
    if target_chapters < 1 and not args.analyze_only:
        parser.error("--chapters 至少为 1，或使用 --full-book 自动推断整本书章数")
    asyncio.run(
        run_imitation_flow(
            args.book_id,
            target_chapters,
            force_analyze=args.force_analyze,
            no_seed=args.no_seed,
            analyze_only=args.analyze_only,
        )
    )


if __name__ == "__main__":
    main()
