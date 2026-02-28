# -*- coding: utf-8 -*-
"""
完整测试：不依赖 API 的单元/集成 + 可选依赖 API 的 analyze/write 流程。
在项目根目录执行：python tests/run_full_test.py
"""
import json
import os
import sys
from pathlib import Path

# 项目根
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 加载 .env，使 Phase 2 能读到 DEEPSEEK_API_KEY / OPENAI_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

DATA_RAW = ROOT / "data" / "raw"
DATA_CARDS = ROOT / "data" / "cards"


def phase1_no_api():
    """阶段一：无需 API 的导入与逻辑测试。"""
    print("\n========== Phase 1: 无 API 依赖 ==========")
    ok = 0
    total = 0

    # 1. 各包导入
    total += 1
    try:
        from src.analyzer import AnalysisState, create_initial_state, load_state, save_state
        from src.writer import WriterState
        from src.user_center import UserProfile, parse_intent, SatisfactionTracker
        from src.librarian.style_store import StyleStore, StyleFingerprint
        from src.evolution import (
            TraceLogger, TraceEvent, SimulatorAI, EvaluationResult,
            apply_patch, apply_edits, save_snapshot, get_next_version,
        )
        from src.evolution.file_operator import extract_paths_from_markdown
        print("  [OK] 各包导入")
        ok += 1
    except Exception as e:
        print(f"  [FAIL] 导入: {e}")
        return ok, total

    # 2. TraceLogger 单例 + get_trace
    total += 1
    try:
        from src.evolution import TraceLogger
        log = TraceLogger()
        log.log_event_sync("full-test-run", "test", inputs={"x": 1}, outputs={"y": 2})
        trace = log.get_trace("full-test-run")
        assert len(trace) >= 1 and trace[0].module == "test"
        print("  [OK] TraceLogger log + get_trace")
        ok += 1
    except Exception as e:
        print(f"  [FAIL] TraceLogger: {e}")

    # 3. EvaluationResult + average_score
    total += 1
    try:
        from src.evolution import SimulatorAI, EvaluationResult
        r = EvaluationResult(style_score=80, logic_score=70, coherence_score=90, critique="test")
        avg = SimulatorAI.average_score(r)
        assert 79 <= avg <= 81
        print("  [OK] EvaluationResult + average_score")
        ok += 1
    except Exception as e:
        print(f"  [FAIL] SimulatorAI: {e}")

    # 4. file_operator extract_paths + apply_patch 安全路径
    total += 1
    try:
        from src.evolution.file_operator import extract_paths_from_markdown, resolve_safe, apply_patch, PROJECT_ROOT
        paths = extract_paths_from_markdown("see src/writer/draft_layer.py and src/analyzer/backtrack.py")
        assert "src/writer/draft_layer.py" in paths and "src/analyzer/backtrack.py" in paths
        bad = resolve_safe(PROJECT_ROOT, "../../etc/passwd")
        assert bad is None
        print("  [OK] file_operator extract_paths + resolve_safe")
        ok += 1
    except Exception as e:
        print(f"  [FAIL] file_operator: {e}")

    # 5. 已有 raw 书籍 JSON 加载 + StyleStore
    total += 1
    try:
        book_id = "7320218217488600126"
        json_path = DATA_RAW / book_id / f"{book_id}.json"
        if not json_path.is_file():
            # 任选一本存在的
            for d in DATA_RAW.iterdir():
                if d.is_dir():
                    j = d / f"{d.name}.json"
                    if j.is_file():
                        json_path = j
                        book_id = d.name
                        break
        if json_path.is_file():
            store = StyleStore(book_id=book_id)
            n = store.load_from_book_json(json_path)
            assert n >= 0
            fp = store.get_fingerprint()
            assert fp is None or (fp.avg_chapter_length >= 0 and fp.dialogue_ratio >= 0)
            print(f"  [OK] StyleStore 加载 {json_path.name} 样本数 {len(store.samples)}")
        else:
            print("  [SKIP] 无 data/raw 书籍 JSON")
        ok += 1
    except Exception as e:
        print(f"  [FAIL] StyleStore: {e}")

    # 6. 提取器解析鲁棒性：_extract_json_from_response / _normalize_extraction_data
    total += 1
    try:
        from src.analyzer.extractor import _extract_json_from_response, _normalize_extraction_data
        # 带 markdown 包裹
        raw1 = '```json\n{"knowledge_cards": [{"type": "人物", "name": "测试"}], "plot_nodes": []}\n```'
        d1 = _extract_json_from_response(raw1)
        assert d1 is not None and len(d1.get("knowledge_cards") or []) == 1
        # 前后有废话，取首尾 {}
        raw2 = '根据正文，输出如下：\n{"knowledge_cards": [], "plot_nodes": [{"id": "n1", "summary": "x"}]}\n以上为结果。'
        d2 = _extract_json_from_response(raw2)
        assert d2 is not None and len(d2.get("plot_nodes") or []) == 1
        # 中文键兼容
        norm = _normalize_extraction_data({"知识卡片": [{"type": "设定", "name": "a"}], "剧情节点": []})
        assert "knowledge_cards" in norm and len(norm["knowledge_cards"]) == 1
        print("  [OK] 提取器解析鲁棒性（markdown/中文键）")
        ok += 1
    except Exception as e:
        print(f"  [FAIL] 提取器解析: {e}")

    # 7. build_novel_database 与 NovelDatabase 结构
    total += 1
    try:
        from src.analyzer import create_initial_state, build_novel_database
        from src.analyzer.models import KnowledgeCard, PlotNode
        state = create_initial_state("test_book", "测试书")
        state.cards = [KnowledgeCard(type="人物", name="张三", description="主角")]
        state.plot_tree = {"n1": PlotNode(id="n1", type="event", summary="开篇")}
        db = build_novel_database(state)
        assert db.book_id == "test_book"
        assert "角色" in db.entities_by_type and len(db.entities_by_type["角色"]) == 1
        assert len(db.plot_tree) == 1
        print("  [OK] build_novel_database 与 NovelDatabase 结构")
        ok += 1
    except Exception as e:
        print(f"  [FAIL] build_novel_database: {e}")

    # 8. StyleFingerprint 写作习惯序列化
    total += 1
    try:
        from src.librarian.style_store import StyleFingerprint
        fp = StyleFingerprint(book_id="b1", writing_habits="短句为主", sentence_style="长短结合")
        d = fp.to_dict()
        assert d.get("writing_habits") == "短句为主"
        fp2 = StyleFingerprint.from_dict(d)
        assert fp2.writing_habits == fp.writing_habits
        print("  [OK] StyleFingerprint 写作习惯序列化")
        ok += 1
    except Exception as e:
        print(f"  [FAIL] StyleFingerprint: {e}")

    # 9. parse_intent 规则模式 (allow_llm=False)
    total += 1
    try:
        from src.user_center import parse_intent
        p = parse_intent("我想续写下一章", allow_llm=False)
        assert p.intent_type in ("continue", "other") or "续写" in p.raw_text
        print("  [OK] parse_intent(allow_llm=False)")
        ok += 1
    except Exception as e:
        print(f"  [FAIL] parse_intent: {e}")

    # 10. Analyzer create_initial_state + WriterState 构造
    total += 1
    try:
        from src.analyzer import create_initial_state
        from src.writer import WriterState
        a = create_initial_state("test-book", "测试书名")
        assert a.book_id == "test-book"
        w = WriterState(book_id="test-book", chapter_index=0, user_intent="续写")
        assert w.preference_patch == "" and w.user_intent == "续写"
        print("  [OK] AnalysisState + WriterState 构造")
        ok += 1
    except Exception as e:
        print(f"  [FAIL] state: {e}")

    # 11. snapshot get_next_version + save_snapshot (写到一个临时目录)
    total += 1
    try:
        from src.evolution import save_snapshot, get_next_version
        snap_dir = ROOT / "snapshots"
        v = get_next_version(snap_dir)
        assert v >= 1
        out = save_snapshot(output_dir=snap_dir, version=99999, avg_score=85.0, note="full_test")
        assert out.is_file() and out.suffix == ".zip"
        print(f"  [OK] snapshot get_next_version + save_snapshot -> {out.name}")
        try:
            out.unlink()
        except Exception:
            pass
        ok += 1
    except Exception as e:
        print(f"  [FAIL] snapshot: {e}")

    return ok, total


def phase1_5_evolution():
    """Phase 1.5：Evolution 优化模块专项（无 API）。"""
    print("\n========== Phase 1.5: Evolution 优化模块 ==========")
    ok = 0
    total = 0

    # 1. file_operator apply_patch：向允许路径写入并读回
    total += 1
    try:
        from src.evolution.file_operator import apply_patch, apply_edits, PROJECT_ROOT, resolve_safe
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="novel_evolve_", dir=ROOT))
        try:
            # 在临时目录下建 src/evolve_test 占位
            (tmp / "src" / "evolve_test").mkdir(parents=True)
            test_file = tmp / "src" / "evolve_test" / "_test_dummy.py"
            test_file.write_text("# evolve test", encoding="utf-8")
            rel = str(test_file.relative_to(tmp))
            success, msg = apply_patch(tmp, rel, "# patched by test\n")
            assert success, msg
            assert "patched" in (tmp / "src" / "evolve_test" / "_test_dummy.py").read_text(encoding="utf-8")
            print("  [OK] file_operator apply_patch 写入并验证")
            ok += 1
        finally:
            import shutil
            if tmp.is_dir():
                shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        print(f"  [FAIL] file_operator apply_patch: {e}")

    # 2. apply_edits 批量（对临时路径）
    total += 1
    try:
        from src.evolution.file_operator import apply_edits, PROJECT_ROOT
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="novel_edits_", dir=ROOT))
        (tmp / "src").mkdir(parents=True)
        f = tmp / "src" / "_edits_test.py"
        f.write_text("old", encoding="utf-8")
        rel = "src/_edits_test.py"
        results = apply_edits(tmp, [{"path": rel, "new_content": "new"}])
        assert len(results) == 1 and results[0][0] is True
        assert f.read_text(encoding="utf-8") == "new"
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        print("  [OK] file_operator apply_edits")
        ok += 1
    except Exception as e:
        print(f"  [FAIL] apply_edits: {e}")

    # 3. TraceLogger save_to_file
    total += 1
    try:
        from src.evolution import TraceLogger
        import tempfile
        log_dir = Path(tempfile.mkdtemp(prefix="novel_logs_", dir=ROOT))
        logger = TraceLogger(log_dir=log_dir)
        logger.log_event_sync("save-test", "test", inputs={}, outputs={})
        p = logger.save_to_file(log_dir / "execution_trace.jsonl")
        assert p.is_file() and p.stat().st_size > 0
        import shutil
        shutil.rmtree(log_dir, ignore_errors=True)
        print("  [OK] TraceLogger save_to_file")
        ok += 1
    except Exception as e:
        print(f"  [FAIL] TraceLogger save_to_file: {e}")

    # 4. EngineeringDiagnostician 实例化 + diagnose 入参类型
    total += 1
    try:
        from src.evolution import EngineeringDiagnostician, EvaluationResult
        from src.evolution.trace_logger import TraceEvent
        diag = EngineeringDiagnostician()
        r = EvaluationResult(style_score=60, logic_score=50, coherence_score=55, critique="测试")
        plan = diag.diagnose(r, [TraceEvent(run_id="x", module="writer", inputs={}, outputs={})], code_context="")
        assert isinstance(plan, str) and len(plan) >= 0
        print("  [OK] EngineeringDiagnostician.diagnose 可调用（需 API 才返回非空）")
        ok += 1
    except Exception as e:
        print(f"  [FAIL] EngineeringDiagnostician: {e}")

    # 5. EvolutionLoop.run_chapter 与 mock generate_fn（仅检查返回结构，不测评分）
    total += 1
    try:
        from src.evolution import EvolutionLoop
        book_id = "7320218217488600126"
        json_path = DATA_RAW / book_id / f"{book_id}.json"
        if not json_path.is_file():
            for d in DATA_RAW.iterdir():
                if d.is_dir():
                    j = d / f"{d.name}.json"
                    if j.is_file():
                        book_id = d.name
                        json_path = j
                        break
        if not json_path.is_file():
            print("  [SKIP] 无 data/raw 书籍，跳过 run_chapter")
        else:
            loop = EvolutionLoop(book_id=book_id, data_raw=DATA_RAW, data_cards=DATA_CARDS)
            def mock_gen(bid, ch):
                return "这是一段假草稿。", "mock-run-id"
            result, run_id, orig = loop.run_chapter(0, mock_gen)
            assert run_id == "mock-run-id"
            assert hasattr(result, "style_score") and hasattr(result, "critique")
            print("  [OK] EvolutionLoop.run_chapter(mock) 返回结构正确")
        ok += 1
    except Exception as e:
        print(f"  [FAIL] EvolutionLoop.run_chapter: {e}")

    return ok, total


def phase2_with_api():
    """阶段二：需要 API Key 的 analyze（2 章）+ write --preview-only。"""
    print("\n========== Phase 2: 依赖 API（analyze + write preview）==========")
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        print("  [SKIP] 未设置 DEEPSEEK_API_KEY / OPENAI_API_KEY，跳过 Phase 2")
        return 0, 0

    ok = 0
    total = 0
    book_id = "7320218217488600126"
    json_path = DATA_RAW / book_id / f"{book_id}.json"
    if not json_path.is_file():
        for d in DATA_RAW.iterdir():
            if d.is_dir():
                j = d / f"{d.name}.json"
                if j.is_file():
                    book_id = d.name
                    json_path = j
                    break
    if not json_path.is_file():
        print("  [SKIP] 无 data/raw 书籍，跳过 Phase 2")
        return 0, 0

    # analyze：智能采样→元知识模板→并发逐章提取→高质量整合→小说数据库+风格指纹
    total += 1
    try:
        from src.analyzer import run_full_pipeline, save_state, build_novel_database
        from src.librarian.style_store import build_style_fingerprint_library
        data = json.loads(json_path.read_text(encoding="utf-8"))
        chapters = (data.get("chapters") or [])[:2]
        title = data.get("title") or book_id
        if len(chapters) < 2:
            print("  [SKIP] 该书章节不足 2 章")
        else:
            state = run_full_pipeline(
                book_id=book_id,
                title=title,
                chapters=chapters,
                total_chapter_count=len(data.get("chapters") or []),
                refine_every_n_windows=1,
                use_per_chapter_extraction=True,
                use_concurrent_extraction=True,
                max_concurrent_workers=2,
            )
            DATA_CARDS.mkdir(parents=True, exist_ok=True)
            out_dir = DATA_CARDS / book_id
            out_dir.mkdir(parents=True, exist_ok=True)
            save_state(state, out_dir / "analysis_state.json")
            novel_db = build_novel_database(state)
            (out_dir / "novel_database.json").write_text(
                novel_db.model_dump_json(exclude_none=False, indent=2), encoding="utf-8"
            )
            build_style_fingerprint_library(book_id, title, json_path, out_dir / "style_fingerprint.json")
            assert (out_dir / "analysis_state.json").is_file()
            assert (out_dir / "novel_database.json").is_file()
            assert (out_dir / "style_fingerprint.json").is_file()
            loaded_state = json.loads((out_dir / "analysis_state.json").read_text(encoding="utf-8"))
            assert "meta_protocol" in loaded_state and "cards" in loaded_state and "plot_tree" in loaded_state
            loaded_db = json.loads((out_dir / "novel_database.json").read_text(encoding="utf-8"))
            assert "entities_by_type" in loaded_db
            for k in ["设定", "道具", "场景", "角色", "事件"]:
                assert k in loaded_db["entities_by_type"], f"novel_database 缺少实体类型 {k}"
            loaded_fp = json.loads((out_dir / "style_fingerprint.json").read_text(encoding="utf-8"))
            assert "book_id" in loaded_fp and ("writing_habits" in loaded_fp or "avg_chapter_length" in loaded_fp)
            print(f"  [OK] analyze 2 章 -> cards {len(state.cards)} 条, 节点 {len(state.plot_tree)} 个; novel_db + style_fp 已产出并校验")
            print(f"      分析模块产出: {out_dir / 'analysis_state.json'}, novel_database.json, style_fingerprint.json")
            ok += 1
    except Exception as e:
        print(f"  [FAIL] analyze: {e}")

    # write --preview-only（仅 3 走向，不生成正文）
    total += 1
    try:
        from src.analyzer import load_state_for_rewrite
        from src.librarian.context_loader import build_rewrite_context
        from src.writer import WriterState, get_three_plot_directions
        analysis_state = load_state_for_rewrite(book_id, DATA_CARDS)
        if not analysis_state:
            print("  [SKIP] 无 analysis_state，跳过 write preview")
        else:
            ctx = build_rewrite_context(
                analysis_state, rewritten_chapter_index=0, new_anchors_text="续写下一章",
            )
            state = WriterState(
                book_id=book_id,
                chapter_index=0,
                user_intent="续写下一章",
                history_causal_pack=ctx["history_causal_pack"],
                new_causal_anchors=ctx["new_causal_anchors"],
                rule_constraints_pack=ctx["rule_constraints_pack"],
            )
            directions = get_three_plot_directions(state)
            assert len(directions) <= 3
            print(f"  [OK] write preview -> {len(directions)} 个走向")
            ok += 1
    except Exception as e:
        print(f"  [FAIL] write preview: {e}")

    return ok, total


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NovelAgent 完整测试")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None, help="仅运行指定阶段: 1 或 2")
    parser.add_argument("--analyzer", action="store_true", help="仅运行与分析模块相关的测试（Phase1 全量 + Phase2）")
    args = parser.parse_args()

    print("NovelAgent 完整测试")
    if args.phase == 2:
        o1, t1, o1_5, t1_5 = 0, 0, 0, 0
        o2, t2 = phase2_with_api()
    elif args.analyzer:
        o1, t1 = phase1_no_api()
        o1_5, t1_5 = phase1_5_evolution()
        o2, t2 = phase2_with_api()
    else:
        o1, t1 = phase1_no_api()
        o1_5, t1_5 = phase1_5_evolution()
        o2, t2 = phase2_with_api() if (args.phase is None or args.phase != 1) else (0, 0)

    total_ok = o1 + o1_5 + o2
    total_all = t1 + t1_5 + t2
    print("\n" + "=" * 50)
    print(f"结果: {total_ok}/{total_all} 通过 (Phase1: {o1}/{t1}, Phase1.5 Evolution: {o1_5}/{t1_5}, Phase2: {o2}/{t2})")
    if total_ok < total_all:
        sys.exit(1)
    print("全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
