# 用 LangGraph Studio 做 UI 检测

本项目已接入 **LangGraph 官方 Studio**，可在浏览器里看到写作图的结构、节点与每步状态，并单步调试。

## 1. 安装 CLI（含 inmem）

```bash
pip install "langgraph-cli[inmem]"
```

或从项目根安装依赖时已包含：

```bash
pip install -r requirements.txt
```

## 2. 启动本地开发服务器 + Studio

在**项目根目录**执行：

```bash
langgraph dev
```

- 会自动读取根目录的 `langgraph.json`
- 启动本地图服务，并打开浏览器连到 **LangGraph Studio**（或提示打开 URL）
- 在 Studio 里选择 **writer** 图，即可看到：
  - 图结构：user_probe → planner → consultant → writer → critique（含条件边）
  - 输入 state 的编辑与运行
  - 每步执行后的状态输出

## 3. 项目中的配置

| 文件 | 作用 |
|------|------|
| `langgraph.json` | 声明图入口：`writer` → `./langgraph_app/writer.py:graph` |
| `langgraph_app/writer.py` | 暴露 `graph = build_writer_graph()`，供 Studio 加载 |

## 4. 注意

- 首次使用可能需要 **LangSmith 账号**（免费注册），用于 Studio 云端会话。
- 若只用本地图、不想连云，可查官方文档是否有 `langgraph dev --local` 等纯本地模式。
- 在 Studio 里输入的 state 需为 **可 JSON 序列化** 的 dict；`_analysis_state`、`_style_injector` 等不可序列化对象在 UI 中不会传入，实际调用 writer 节点时会从磁盘按 `book_id` 加载。

## 5. 参考

- [LangGraph Studio 本地连接](https://langchain-ai.github.io/langgraph/how-tos/local-studio/)
- [Run a LangGraph app locally](https://docs.langchain.com/langgraph-platform/local-server)
