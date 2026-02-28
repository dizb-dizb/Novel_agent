---
description: 强制执行 LangChain 和 LangGraph 的开发规范
globs: src/**/*.py
---

# LangChain 开发规范

- **优先使用 LCEL (LangChain Expression Language)**: 
  - 组合逻辑必须使用 `|` 符号。
  - 示例: `chain = prompt | model | parser`。
- **状态管理 (LangGraph)**: 
  - 必须使用 `TypedDict` 或 Pydantic 定义 `State` 类。
  - 所有图节点 (Nodes) 必须返回 `State` 的部分更新。
- **结构化输出**: 
  - 优先使用 `llm.with_structured_output(PydanticModel)`。
- **异步支持**: 
  - 涉及网络 IO（爬虫、API 调用）的操作必须使用 `async` / `await`。