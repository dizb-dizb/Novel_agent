# Novel-Agent

基于「分析 → 存储 → 续写」流程的小说智能助手：从 `data/raw` 下的书籍 JSON 做知识抽取与剧情树构建，支持续写、改写、逻辑回溯与自我改良优化。

## 功能概览

| 命令 | 说明 |
|------|------|
| `analyze` | 整书智能采样 → 元知识模板 → 并发逐章提取 → 高质量整合 → 小说数据库与风格指纹 |
| `full-pipeline` | 一键流水线：分析 → 写库 → 仿写（可指定仿写章节数或逆向重构） |
| `backtrack` | 逻辑回溯：检查某剧情是否与既有知识卡片一致 |
| `rewrite` | 改写第 N 章：影响评估、三维上下文、逻辑+风格双重校对 |
| `write` | 续写生成：三维上下文 + 3 走向 → 选分支 → 分层生成（战略→逻辑→草稿→润色） |
| `evolve` | 自我改良：续写→评分→不合格则诊断并改码→重跑，直到达标或达最大迭代 |
| `user-profile` | 根据描述生成用户偏好协议，供续写时注入 |

## 环境要求

- Python 3.10+
- 需配置 LLM API Key（见下方「环境变量」）

## 安装

```bash
# 克隆后进入项目目录
cd "NovelAgent - public"

# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# 安装依赖
pip install -r requirements.txt
```

## 环境变量

在项目根目录创建 `.env`，按需配置（密钥不要提交到仓库）：

```env
# 推理/续写（至少配置其一）
DEEPSEEK_API_KEY=sk-xxx
OPENAI_API_KEY=sk-xxx

# 文笔/分析（可选）
DASHSCOPE_API_KEY=xxx
MOONSHOT_API_KEY=xxx
ZHIPU_API_KEY=xxx

# 分析长上下文模型（可选，见 src/utils/llm_client.py）
ANALYZER_LONG_CONTEXT_MODEL=kimi-k2-turbo-preview
```

## 数据目录约定

- **data/raw/{book_id}/** — 原始书籍数据，需包含 `{book_id}.json`（含 `title`、`chapters`，每章含 `chapter_title`、`content`）
- **data/cards/{book_id}/** — 分析产出：`analysis_state.json`、`novel_database.json`、`style_fingerprint.json` 等
- **data/user_center/** — 用户偏好协议等

书籍 JSON 需自行准备或由其他工具生成后放入 `data/raw/{book_id}/`。

### raw 书籍 JSON 样例

目录：`data/raw/{book_id}/{book_id}.json`，例如 `data/raw/my_novel/my_novel.json`：

```json
{
  "title": "示例书名",
  "chapters": [
    {
      "chapter_id": "ch_1",
      "chapter_title": "第一章 开端",
      "content": "这是第一章的正文内容。\n\n可以多段，用空行分隔。分析、续写、风格指纹等都会从这里读取。"
    },
    {
      "chapter_id": "ch_2",
      "chapter_title": "第二章 转折",
      "content": "第二章正文……"
    }
  ]
}
```

- 顶层必填：`title`（书名）、`chapters`（章节数组）。
- 每章必填：`chapter_title`、`content`；`chapter_id` 可选（不写时按索引处理）。

## 使用示例

```bash
# 分析：对 data/raw/{book_id} 做智能采样 + 元协议 + 逐章提取 + 整合
python main.py analyze --book-id 7320218217488600126
python main.py analyze --book-id 7320218217488600126 --chapters 30

# 逻辑回溯（写作前检查剧情是否合理）
python main.py backtrack --book-id 7320218217488600126 --action "张三在此时杀掉李四"

# 续写：三维上下文 + 3 走向 → 选分支 → 分层生成
python main.py write --book-id 7320218217488600126 --chapter 10 --intent "主角潜入敌营发现密信"
python main.py write --book-id 7320218217488600126 --chapter 10 --intent "..." --branch 1 --out chapter_11_draft.txt
python main.py write --book-id 7320218217488600126 --chapter 10 --intent "..." --preview-only

# 自我改良：续写→评分→不合格则诊断改码→重跑
python main.py evolve --book-id 7320218217488600126 --chapter 0 --threshold 85 --max-iter 10

# 一键流水线：分析 → 写库 → 仿写（示例：仿写 2 章）
python main.py full-pipeline --book-id 7320218217488600126 --imitation-chapters 2
```

更多参数见 `python main.py <command> --help`。

## 测试

```bash
# 完整测试（Phase1 无 API + Phase1.5 Evolution + Phase2 依赖 API）
python tests/run_full_test.py

# 仅 Phase 1
python tests/run_full_test.py --phase 1
```

## 项目结构（简要）

```
├── main.py              # 入口与子命令
├── requirements.txt
├── data/
│   ├── raw/             # 书籍 JSON，按 book_id 分子目录
│   ├── cards/           # 分析结果与知识库
│   └── user_center/     # 用户偏好等
├── src/
│   ├── analyzer/        # 元协议、抽取、剧情树、回溯
│   ├── writer/          # 续写、改写、分层生成
│   ├── librarian/       # 风格库、上下文加载
│   ├── user_center/     # 用户偏好解析与追踪
│   ├── evolution/       # 评分、诊断、改码、快照
│   └── utils/           # 日志、LLM 客户端等
├── backend/             # 可选：仿写流水线、数据库等
└── tests/
```
##
生成的初次样本为example.txt
## 许可证

按项目仓库约定使用。
