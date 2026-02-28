# 自我改良优化模块（Evolution）

本模块实现「续写 → 评分 → 不合格则诊断并自动改码 → 再跑」的闭环，直到评分达标或达到最大迭代；达标时自动打快照备份。

---

## 1. 模块位置与组成

| 路径 | 作用 |
|------|------|
| **src/evolution/trace_logger.py** | 黑匣子：记录每次生成的 prompt、context、输出、报错 |
| **src/evolution/simulator.py** | 裁判：原著 vs AI 续写 盲测打分（文风/逻辑/剧情推动力） |
| **src/evolution/engineering.py** | 医生：根据低分 + trace 产出 Markdown Action Plan（问题分析 + 改码建议） |
| **src/evolution/snapshot.py** | 档案员：把 src/ 与 .cursor/rules 打成 zip 快照 |
| **src/evolution/runner.py** | 运行器：读原著→生成续写→评分，低分可暂停等人改（Human-in-the-loop） |
| **src/evolution/file_operator.py** | 文件操作员：按 LLM 给出的编辑安全写回 src/ 下的代码 |
| **src/evolution/orchestrator.py** | 主控：循环执行「跑章节→评分→不合格则诊断→plan_to_edits→改码→再跑」，达标则快照 |

---

## 2. 一键运行（推荐）

在项目根目录执行：

```bash
python main.py evolve --book-id 7320218217488600126 --chapter 0 --threshold 85 --max-iter 10
```

- **--book-id**：必填，用于读 `data/raw/{book_id}` 与 `data/cards/{book_id}`（需先对该书跑过 `analyze`）。
- **--chapter**：测试章节索引（0-based），默认 0。
- **--threshold**：达标分数（三维平均），默认 85。
- **--max-iter**：最大迭代次数，默认 10。
- **--out-dir / --raw-dir**：可指定 data/cards、data/raw 路径。

流程简述：对指定章节做续写 → Simulator 打分 → 若平均分 &lt; threshold，则 EngineeringDiagnostician 出 Action Plan → plan_to_edits 生成补丁 → file_operator 改源码 → 下一轮；若 ≥ threshold 则 save_snapshot 并结束。

---

## 3. 其他入口

| 用途 | 命令或调用 |
|------|------------|
| 只应用已有补丁（不跑评分） | `python -m src.evolution.apply_patch patches/xxx.json` |
| 主控环（不通过 main.py） | `python -m src.evolution.orchestrator --book-id xxx --chapter 0 --threshold 85 --max-iter 10` |
| 人机在环（低分暂停等人改，不自动改码） | 代码中 `EvolutionLoop(..., on_pause=...)` 后调用 `loop.run(...)` |

---

## 4. 数据与快照

- **Trace 日志**：`data/logs/execution_trace.jsonl`（需在流程中调用 `TraceLogger().save_to_file()`）。
- **快照**：达标时写入 `snapshots/v{version}_score_{avg_score}.zip`，内含 src/、.cursor/rules 与 version_report.md。

---

## 5. .cursor 规则

已配置 `.cursor/rules/evolution_auto_driver.md`：需要优化架构时，优先通过 `apply_patch` 或 `orchestrator` 自动改码，而不是仅在对话中给出修改建议。
