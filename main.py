# -*- coding: utf-8 -*-
"""

用法示例：
  # 分析：对 data/raw/{book_id} 的正文做「智能采样 + 元协议 + 滑动窗口提取 + 整合」
  python main.py analyze --book-id 7320218217488600126
  python main.py analyze --book-id 7320218217488600126 --chapters 30

  # 逻辑回溯（写作前检查剧情是否合理）
  python main.py backtrack --book-id 7320218217488600126 --action "张三在此时杀掉李四"

  # 续写/改写生成：三维上下文 + 3 走向 → 选分支 → 分层生成（战略→逻辑→草稿→润色）
  python main.py write --book-id 7320218217488600126 --chapter 10 --intent "主角潜入敌营发现密信"
  python main.py write --book-id 7320218217488600126 --chapter 10 --intent "..." --branch 1 --out chapter_11_draft.txt
  python main.py write --book-id 7320218217488600126 --chapter 10 --intent "..." --preview-only

  # 自我改良优化：续写→评分→不合格则诊断并自动改码→再跑，直到达标或达最大迭代
  python main.py evolve --book-id 7320218217488600126 --chapter 0 --threshold 85 --max-iter 10
"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# 项目根
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_CARDS = ROOT / "data" / "cards"
DATA_USER_CENTER = ROOT / "data" / "user_center"


def cmd_analyze(args):
    """分析：整书智能采样 → 元知识模板（长上下文）→ 低质量模型逐章并发提取 → 高质量整合 → 小说数据库。"""
    from src.analyzer import build_novel_database, load_state, run_full_pipeline, run_phase2_only, save_state
    from src.utils import get_logger

    log = get_logger()
    book_id = getattr(args, "book_id", "") or ""
    if not book_id:
        log.error("请指定 --book-id")
        return
    raw_dir = Path(args.raw_dir) if (getattr(args, "raw_dir", "") and str(args.raw_dir).strip()) else DATA_RAW
    cards_dir = Path(args.out_dir) if (getattr(args, "out_dir", "") and str(args.out_dir).strip()) else DATA_CARDS
    book_dir = raw_dir / book_id
    json_path = book_dir / f"{book_id}.json"
    if not json_path.is_file():
        log.error("未找到书籍 JSON: %s", json_path)
        return
    import json
    data = json.loads(json_path.read_text(encoding="utf-8"))
    chapters = data.get("chapters") or []
    title = data.get("title") or book_id
    max_chapters = getattr(args, "chapters", None)
    if max_chapters is not None:
        chapters = chapters[:max_chapters]
    if not chapters:
        log.warning("无章节可分析")
        return
    re_extract = getattr(args, "re_extract", False)
    if re_extract:
        out_book = cards_dir / book_id
        state_path = out_book / "analysis_state.json"
        state = load_state(state_path)
        if not state or not state.meta_protocol:
            log.error("未找到已有分析状态或元协议，请先执行完整 analyze（无 --re-extract）: %s", state_path)
            return
        log.info("重新执行章节提取（沿用已有元协议）: %s, 共 %s 章", title, len(chapters))
        state = run_phase2_only(
            state,
            chapters,
            use_per_chapter_extraction=True,
            use_concurrent_extraction=not getattr(args, "no_concurrent", False),
            max_concurrent_workers=getattr(args, "workers", 4),
            consolidate_every_n_chapters=1,
        )
    else:
        log.info("开始分析: %s, 共 %s 章（智能采样→元知识模板→并发逐章提取→高质量整合）", title, len(chapters))
        state = run_full_pipeline(
            book_id=book_id,
            title=title,
            chapters=chapters,
            total_chapter_count=len(data.get("chapters") or []),
            refine_every_n_windows=getattr(args, "refine_every", 1),
            use_concurrent_extraction=not getattr(args, "no_concurrent", False),
            max_concurrent_workers=getattr(args, "workers", 4),
        )
    out_book = cards_dir / book_id
    out_book.mkdir(parents=True, exist_ok=True)
    state_path = out_book / "analysis_state.json"
    save_state(state, state_path)
    novel_db = build_novel_database(state)
    db_path = out_book / "novel_database.json"
    db_path.write_text(novel_db.model_dump_json(exclude_none=False, indent=2), encoding="utf-8")
    from src.librarian.style_store import build_style_fingerprint_library
    fp_path = out_book / "style_fingerprint.json"
    build_style_fingerprint_library(book_id, title, json_path, fp_path)
    log.info("分析完成: 卡片 %s 个, 节点 %s 个, 冲突 %s 个; 状态已保存至 %s, 小说数据库至 %s, 风格指纹库至 %s",
             len(state.cards), len(state.plot_tree), len(state.conflict_marks), state_path, db_path, fp_path)
    if getattr(args, "write_db", False):
        from src.analyzer.write_to_db import write_novel_database_to_backend
        result = write_novel_database_to_backend(book_id, db_path)
        if result.get("ok"):
            log.info("已写入后端数据库: %s", result.get("written"))
        else:
            log.warning("写入后端数据库失败: %s", result.get("error"))


def cmd_rewrite(args):
    """改写流：加载 state → 影响评估 → 三维上下文 → 可选双重校对。"""
    from src.analyzer import load_state_for_rewrite, run_rewrite_flow
    from src.utils import get_logger

    log = get_logger()
    book_id = getattr(args, "book_id", "") or ""
    chapter_index = getattr(args, "chapter", 0)
    anchors = getattr(args, "anchors", "") or ""
    draft = getattr(args, "draft", "") or ""
    cards_dir = Path(args.out_dir) if (getattr(args, "out_dir", "") and str(args.out_dir).strip()) else DATA_CARDS

    state = load_state_for_rewrite(book_id, cards_dir)
    if not state:
        log.error("未找到分析状态，请先运行: python main.py analyze --book-id %s", book_id)
        return
    result = run_rewrite_flow(
        state,
        chapter_index=chapter_index,
        new_anchors_text=anchors,
        draft_text_for_check=draft,
    )
    log.info("影响报告: %s", result["impact_report"].summary)
    log.info("三维上下文已构建（history_causal_pack / new_causal_anchors / rule_constraints_pack）")
    if result.get("double_check"):
        dc = result["double_check"]
        log.info("双重校对: passed=%s, logic=%s, style=%s", dc["passed"], dc["logic"].get("passed"), dc["style"].get("passed"))
        if not dc["passed"] and dc.get("suggestion"):
            log.info("建议: %s", dc["suggestion"])


def cmd_write(args):
    """续写/改写生成：加载分析状态与三维上下文，运行 Planner→Consultant→Writer→Critique（可选仅预览 3 走向）。"""
    from src.analyzer import load_state_for_rewrite
    from src.librarian.context_loader import build_rewrite_context
    from src.writer import (
        WriterState,
        get_three_plot_directions,
        generate_chapter_for_branch,
        run_writer_flow,
        build_writer_graph,
    )
    from src.librarian.style_store import StyleStore
    from src.writer.style_injector import StyleInjector
    from src.utils import get_logger

    log = get_logger()
    book_id = getattr(args, "book_id", "") or ""
    chapter_index = getattr(args, "chapter", 0)
    intent = getattr(args, "intent", "") or ""
    branch = getattr(args, "branch", 0)
    preview_only = getattr(args, "preview_only", False)
    out_path = getattr(args, "out", "") or ""
    max_rounds = getattr(args, "max_rounds", 3)
    cards_dir = Path(args.out_dir) if (getattr(args, "out_dir", "") and str(args.out_dir).strip()) else DATA_CARDS
    raw_dir = Path(args.raw_dir) if (getattr(args, "raw_dir", "") and str(args.raw_dir).strip()) else DATA_RAW

    if not book_id:
        log.error("请指定 --book-id")
        return
    analysis_state = load_state_for_rewrite(book_id, cards_dir)
    if not analysis_state:
        log.error("未找到分析状态，请先运行: python main.py analyze --book-id %s", book_id)
        return

    # 三维上下文（续写起点用 intent 作为新因果锚点摘要）
    ctx = build_rewrite_context(
        analysis_state,
        rewritten_chapter_index=chapter_index,
        new_anchors_text=intent or "（续写下一章，请保持与前文因果一致）",
    )
    state = WriterState(
        book_id=book_id,
        chapter_index=chapter_index,
        user_intent=intent or "续写下一章",
        history_causal_pack=ctx["history_causal_pack"],
        new_causal_anchors=ctx["new_causal_anchors"],
        rule_constraints_pack=ctx["rule_constraints_pack"],
        selected_branch_index=branch,
    )

    # 用户偏好：从 data/user_center/{book_id}_preference.json 加载（可选）
    try:
        from src.user_center import load_preference_protocol, produce_user_context
        pref_path = DATA_USER_CENTER / f"{book_id}_preference.json"
        protocol = load_preference_protocol(pref_path)
        if protocol and protocol.profile:
            uctx = produce_user_context(
                protocol.profile,
                book_id=book_id,
                chapter_index=chapter_index,
            )
            state.preference_patch = uctx.preference_patch
            state.steering_hint = uctx.steering_hint
            log.info("已注入用户偏好补丁（来自 %s）", pref_path.name)
    except Exception:
        pass

    # 风格指纹库：从 data/raw 加载样本，若有 data/cards/{book_id}/style_fingerprint.json 则加载写作手法与习惯
    style_store = StyleStore(book_id=book_id)
    book_json = raw_dir / book_id / f"{book_id}.json"
    fingerprint_file = cards_dir / book_id / "style_fingerprint.json"
    if book_json.is_file():
        style_store.load_from_book_json(book_json, fingerprint_file=fingerprint_file)
        log.info("已加载风格样本: %s 条%s", len(style_store.samples), "，已加载写作风格指纹" if (style_store.fingerprint and (style_store.fingerprint.writing_habits or style_store.fingerprint.sentence_style)) else "")
    style_injector = StyleInjector(style_store=style_store)

    if preview_only:
        directions = get_three_plot_directions(state)
        state.plot_directions = directions
        log.info("三个可选走向（请用 --branch 0/1/2 选择后生成）：")
        for i, d in enumerate(directions):
            log.info("  走向 %s: %s", i, (d.get("summary") or "")[:200])
        return

    if build_writer_graph is None or run_writer_flow is None:
        log.warning("未安装 langgraph，改用单分支生成（无 Consultant/Critique 循环）")
        directions = get_three_plot_directions(state)
        state.plot_directions = directions
        if branch < 0 or branch >= len(directions):
            branch = 0
        state = generate_chapter_for_branch(state, branch, analysis_state, style_injector)
    else:
        state.plot_directions = get_three_plot_directions(state)
        state.selected_branch_index = max(0, min(branch, 2))
        state = run_writer_flow(state, analysis_state, style_injector, max_rounds=max_rounds)

    log.info("章节类型: %s", state.chapter_type)
    log.info("批判通过: %s", state.critique_passed)
    if state.polish_feedback:
        log.info("润色反馈: %s", state.polish_feedback[:200])
    if out_path:
        Path(out_path).write_text(state.draft or "", encoding="utf-8")
        log.info("草稿已写入: %s", out_path)
    else:
        log.info("草稿长度: %s 字", len(state.draft or ""))


def cmd_evolve(args):
    """自我改良优化：运行 evolution 主控环（续写→Simulator 评分→低分则诊断+自动改码→循环），达标时自动快照。"""
    from src.evolution import run_evolution_loop
    from src.utils import get_logger

    log = get_logger()
    book_id = getattr(args, "book_id", "") or ""
    chapter_index = getattr(args, "chapter", 0)
    threshold = getattr(args, "threshold", 85.0)
    max_iter = getattr(args, "max_iter", 10)
    cards_dir = Path(args.out_dir) if (getattr(args, "out_dir", "") and str(args.out_dir).strip()) else DATA_CARDS
    raw_dir = Path(args.raw_dir) if (getattr(args, "raw_dir", "") and str(args.raw_dir).strip()) else DATA_RAW

    if not book_id:
        log.error("请指定 --book-id")
        return
    try:
        passed, score, iterations = run_evolution_loop(
            book_id=book_id,
            chapter_index=chapter_index,
            score_threshold=threshold,
            max_iterations=max_iter,
            data_raw=raw_dir,
            data_cards=cards_dir,
            on_iteration=lambda i, s, m: log.info("[iter %s] score=%.1f %s", i, s, (m or "")[:100]),
        )
        log.info("evolve 结束: passed=%s, best_score=%.1f, iterations=%s", passed, score, iterations)
    except Exception as e:
        log.exception("evolve 失败: %s", e)


def cmd_user_profile(args):
    """用户偏好种子：根据对话或一句描述生成 User_Preference_Protocol.json，写入 data/user_center/{book_id}_preference.json。"""
    from src.user_center import (
        build_profile_from_dialogue,
        build_protocol_from_profile,
        save_preference_protocol,
        produce_user_context,
    )
    from src.utils import get_logger

    log = get_logger()
    book_id = getattr(args, "book_id", "") or ""
    prompt = getattr(args, "prompt", "") or ""
    user_id = getattr(args, "user_id", "") or ""

    if not book_id:
        log.error("请指定 --book-id")
        return
    messages = []
    if prompt:
        messages = [
            {"role": "user", "content": prompt},
        ]
    profile = build_profile_from_dialogue(messages, book_id=book_id, user_id=user_id)
    protocol = build_protocol_from_profile(profile, raw_answers={"prompt": prompt})
    out_dir = Path(args.out_dir) if (getattr(args, "out_dir", "") and str(args.out_dir).strip()) else DATA_USER_CENTER
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{book_id}_preference.json"
    save_preference_protocol(protocol, path)
    log.info("已保存用户偏好协议: %s", path)
    uctx = produce_user_context(profile, book_id=book_id)
    if uctx.preference_patch:
        log.info("偏好补丁预览: %s", uctx.preference_patch[:200])


def cmd_backtrack(args):
    """逻辑回溯：检查「在此时安排某剧情」是否与既有知识卡片一致。"""
    from src.analyzer import load_state, check_consistency
    from src.utils import get_logger

    log = get_logger()
    book_id = getattr(args, "book_id", "") or ""
    action = getattr(args, "action", "") or ""
    if not book_id or not action:
        log.error("请指定 --book-id 与 --action")
        return
    cards_dir = Path(args.out_dir) if (getattr(args, "out_dir", "") and str(args.out_dir).strip()) else DATA_CARDS
    state_path = cards_dir / book_id / "analysis_state.json"
    state = load_state(state_path)
    if not state:
        log.error("未找到分析状态: %s，请先运行 analyze", state_path)
        return
    result = check_consistency(state, action, at_chapter_index=getattr(args, "chapter", None))
    log.info("是否合理: %s", result["is_plausible"])
    if result["conflicts"]:
        log.info("冲突: %s", result["conflicts"])
    if result["suggestion"]:
        log.info("建议: %s", result["suggestion"])


def cmd_full_pipeline(args):
    """一键流水线：分析（全书）→ 写库 → 仿写（全书或指定章数）。用于测试从分析到整本书仿写。"""
    import asyncio
    from src.utils import get_logger

    log = get_logger()
    book_id = getattr(args, "book_id", "") or ""
    if not book_id:
        log.error("请指定 --book-id")
        return
    raw_dir = Path(args.raw_dir) if (getattr(args, "raw_dir", "") and str(args.raw_dir).strip()) else DATA_RAW
    cards_dir = Path(args.out_dir) if (getattr(args, "out_dir", "") and str(args.out_dir).strip()) else DATA_CARDS
    skip_analyze = getattr(args, "skip_analyze", False)
    force_analyze = getattr(args, "force_analyze", False)
    imitation_chapters = getattr(args, "imitation_chapters", None)

    if not skip_analyze:
        log.info("======== 流水线 Phase 0：分析（全书） + 写库 ========")
        analyze_args = argparse.Namespace(
            book_id=book_id,
            chapters=None,
            raw_dir=str(raw_dir) if raw_dir else "",
            out_dir=str(cards_dir) if cards_dir else "",
            refine_every=1,
            workers=getattr(args, "workers", 4),
            no_concurrent=False,
            re_extract=False,
            write_db=True,
        )
        cmd_analyze(analyze_args)

    use_reconstruction = getattr(args, "reconstruction", False)
    if use_reconstruction:
        from backend.engine.run_imitation_test import run_reconstruction_flow
        recon_chapters = getattr(args, "imitation_chapters", None)
        log.info(
            "======== 流水线 Phase 1：整体搭建知识框架 → Phase 2：按框架文本实现 ========"
        )
        asyncio.run(
            run_reconstruction_flow(
                book_id,
                max_chapters=recon_chapters,
                force_analyze=False,
                use_chapter_word_count=True,
                skip_framework_build=getattr(args, "skip_framework", False),
                use_concurrent=getattr(args, "concurrent", False),
                render_workers=getattr(args, "render_workers", 3),
            )
        )
        log.info("流水线结束: 分析→写库→整体搭建知识框架→逐章文本实现 已完成")
        return

    from backend.engine.run_imitation_test import get_book_total_chapters, run_imitation_flow
    total = get_book_total_chapters(book_id, raw_dir, cards_dir)
    if imitation_chapters is not None:
        target_chapters = min(imitation_chapters, total) if total else imitation_chapters
        log.info("仿写目标: %s 章（由 --imitation-chapters 指定）", target_chapters)
    else:
        target_chapters = total
        if target_chapters < 1:
            log.error("无法获取全书章数，请先完成分析或指定 --imitation-chapters")
            return
        log.info("仿写目标: 整本书 %s 章", target_chapters)

    log.info("======== 流水线 Phase 1：仿写 ========")
    asyncio.run(
        run_imitation_flow(
            book_id,
            target_chapters,
            force_analyze=force_analyze,
            no_seed=False,
            analyze_only=False,
        )
    )
    log.info("流水线结束: 分析→写库→仿写 已完成")


def main():
    parser = argparse.ArgumentParser(description="Novel-Agent: 分析-存储-续写")
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = sub.add_parser("analyze", help="整书智能采样→元知识模板→并发逐章提取→高质量整合→小说数据库")
    p_analyze.add_argument("--book-id", required=True, help="data/raw 下的书籍 id")
    p_analyze.add_argument("--chapters", type=int, default=None, help="只分析前 N 章（默认全部）")
    p_analyze.add_argument("--raw-dir", default="", help="原始数据目录，默认 data/raw")
    p_analyze.add_argument("--out-dir", default="", help="输出目录，默认 data/cards")
    p_analyze.add_argument("--refine-every", type=int, default=1, help="窗口模式下每 N 窗做一次整合")
    p_analyze.add_argument("--workers", type=int, default=4, help="并发逐章提取的线程数，默认 4")
    p_analyze.add_argument("--no-concurrent", action="store_true", help="关闭并发，改为逐章串行提取")
    p_analyze.add_argument("--re-extract", action="store_true", help="仅重新执行章节提取，沿用已有元协议（需先跑过完整 analyze）")
    p_analyze.add_argument("--write-db", action="store_true", help="分析完成后将 novel_database 写入后端数据库（角色表+情节树表）")
    p_analyze.set_defaults(func=cmd_analyze)

    # full-pipeline（分析→写库→仿写，一键测试整本书流程）
    p_full = sub.add_parser(
        "full-pipeline",
        help="一键流水线：分析（全书）→ 写库 → 仿写（整本书或 --imitation-chapters N）；用于测试从分析到整本书仿写",
    )
    p_full.add_argument("--book-id", required=True, help="书籍 ID")
    p_full.add_argument("--skip-analyze", action="store_true", help="跳过分析，直接使用已有知识库做仿写")
    p_full.add_argument("--force-analyze", action="store_true", help="与跳过分析互斥；不跳过时强制重新分析")
    p_full.add_argument("--imitation-chapters", type=int, default=None, help="仿写/重构章节数（默认整本书或重构时 2）；测试时可设小值如 2")
    p_full.add_argument("--reconstruction", action="store_true", help="逆向重构模式：从知识库情节树做逻辑适配→渲染→固化，完成类似书籍编写")
    p_full.add_argument("--skip-framework", action="store_true", help="跳过 Phase 1，直接使用已有 reconstructed_outline.json 做 Phase 2 渲染")
    p_full.add_argument("--concurrent", action="store_true", help="Phase 2 多打字机并行渲染 + 后台入库")
    p_full.add_argument("--render-workers", type=int, default=3, help="并行渲染 worker 数（默认 3）")
    p_full.add_argument("--raw-dir", default="", help="原始数据目录，默认 data/raw")
    p_full.add_argument("--out-dir", default="", help="输出目录，默认 data/cards")
    p_full.add_argument("--workers", type=int, default=4, help="分析阶段并发数")
    p_full.set_defaults(func=cmd_full_pipeline)

    # backtrack
    p_backtrack = sub.add_parser("backtrack", help="逻辑回溯：检查某剧情是否与既有卡片一致")
    p_backtrack.add_argument("--book-id", required=True)
    p_backtrack.add_argument("--action", required=True, help="例如：A 在此时杀掉 B")
    p_backtrack.add_argument("--chapter", type=int, default=None, help="剧情发生章节序号 0-based")
    p_backtrack.add_argument("--out-dir", default="", help="data/cards 目录")
    p_backtrack.set_defaults(func=cmd_backtrack)

    # rewrite（改写影响评估 + 三维上下文 + 可选双重校对）
    p_rewrite = sub.add_parser("rewrite", help="改写第 N 章：影响评估、三维上下文、双重校对")
    p_rewrite.add_argument("--book-id", required=True, help="书籍 id")
    p_rewrite.add_argument("--chapter", type=int, required=True, help="被改写的章节序号（0-based）")
    p_rewrite.add_argument("--anchors", default="", help="改写后的核心变数摘要（新因果锚点）")
    p_rewrite.add_argument("--draft", default="", help="续写稿正文（用于逻辑+风格双重校对）")
    p_rewrite.add_argument("--out-dir", default="", help="data/cards 目录")
    p_rewrite.set_defaults(func=cmd_rewrite)

    # write（续写/改写生成：3 走向 → 选分支 → 分层生成，可选保存草稿）
    p_write = sub.add_parser("write", help="续写生成：三维上下文 + 3 走向 → 分层生成（战略→逻辑→草稿→润色）")
    p_write.add_argument("--book-id", required=True, help="书籍 id")
    p_write.add_argument("--chapter", type=int, required=True, help="续写章节序号（0-based，即第 chapter+1 章）")
    p_write.add_argument("--intent", default="", help="续写/改写意图，如：主角潜入敌营发现密信")
    p_write.add_argument("--branch", type=int, default=0, choices=[0, 1, 2], help="选定走向 0/1/2，默认 0")
    p_write.add_argument("--preview-only", action="store_true", help="仅输出 3 个走向摘要，不生成正文")
    p_write.add_argument("--out", default="", help="将草稿写入该文件")
    p_write.add_argument("--max-rounds", type=int, default=3, help="Critique 未通过时最多重试轮数")
    p_write.add_argument("--out-dir", default="", help="data/cards 目录")
    p_write.add_argument("--raw-dir", default="", help="data/raw 目录（用于加载风格样本）")
    p_write.set_defaults(func=cmd_write)

    # evolve（自我改良优化：续写→评分→诊断→自动改码→循环，达标快照）
    p_evolve = sub.add_parser("evolve", help="自我改良优化：评分不合格则诊断并改码后重跑，直到达标或达最大迭代")
    p_evolve.add_argument("--book-id", required=True, help="书籍 id")
    p_evolve.add_argument("--chapter", type=int, default=0, help="测试章节索引 0-based")
    p_evolve.add_argument("--threshold", type=float, default=85.0, help="达标分数")
    p_evolve.add_argument("--max-iter", type=int, default=10, help="最大迭代次数")
    p_evolve.add_argument("--out-dir", default="", help="data/cards 目录")
    p_evolve.add_argument("--raw-dir", default="", help="data/raw 目录")
    p_evolve.set_defaults(func=cmd_evolve)

    # user-profile（用户偏好种子 → User_Preference_Protocol.json）
    p_profile = sub.add_parser("user-profile", help="根据对话/描述生成用户偏好协议，供 write 时注入")
    p_profile.add_argument("--book-id", required=True, help="锚定书籍 id")
    p_profile.add_argument("--prompt", default="", help="用户一句描述，如：我喜欢快节奏、绝境反杀")
    p_profile.add_argument("--user-id", default="", help="可选用户 id")
    p_profile.add_argument("--out-dir", default="", help="输出目录，默认 data/user_center")
    p_profile.set_defaults(func=cmd_user_profile)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
