# Evolution 自动驾驶与自我修改

## 自动优化闭环

当需要优化系统架构或生成/检索逻辑时，应优先通过 **evolution 自动驾驶环** 完成，而不是仅在对话中给出修改建议、等待人工改码。

1. **补丁应用**：若已生成具体代码补丁（如 JSON 编辑列表），请直接编写并执行：
   ```bash
   python -m src.evolution.apply_patch <patch_file.json>
   ```
   或生成 `patches/xxx.json` 后运行上述命令，由 `file_operator` 安全写回 `src/` 或 `.cursor/rules` 下的文件。

2. **全自动优化环**：若希望由「运行 → 评分 → 诊断 → 改码」循环自动跑通，请执行：
   ```bash
   python -m src.evolution.orchestrator --book-id <BOOK_ID> --chapter 0 --threshold 85 --max-iter 10
   ```
   脚本会反复运行续写评测，不合格时调用工程诊断 AI 生成 Action Plan、转为编辑并写回源码，直到评分达标或达到最大迭代次数；达标时会自动调用 snapshot 备份。

3. **权限**：允许 AI 在项目内运行终端命令、读写 `src/` 与 `.cursor/rules`，以便执行 `apply_patch` 与 `orchestrator`。
