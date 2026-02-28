# 网文生成 AI Agent - 数据库交互后端

FastAPI + SQLAlchemy (SQLite) + Neo4j，分层结构。

## 项目目录树

```
backend/
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI 入口、lifespan 初始化 DB
├── models/                   # SQLAlchemy ORM 模型
│   └── __init__.py
├── schemas/                  # Pydantic 请求/响应模式
│   └── __init__.py
├── database/                 # 数据库配置
│   ├── __init__.py
│   └── database.py           # SQLAlchemy 异步引擎 + Neo4j 驱动
├── services/                 # 业务逻辑层
│   └── __init__.py
├── routers/                  # API 路由
│   └── __init__.py
├── requirements.txt
└── README.md
```

## 环境

- Python 3.10+
- SQLite：自动创建于 `../data/backend.db`（可配置 `SQLITE_PATH`）
- Neo4j：本地默认 `bolt://localhost:7687`（可配置 `NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD`）

## 安装与运行

```bash
# 在项目根目录 NovelAgent 下
cd backend
pip install -r requirements.txt

# 从 backend 目录启动（推荐）
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 或从项目根目录启动
cd ..
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

健康检查：`GET http://localhost:8000/health`

## database.py 说明

- **SQLAlchemy**：异步引擎 `sqlite+aiosqlite`，`AsyncSessionLocal` 会话工厂，`get_async_session()` 供依赖注入。
- **Neo4j**：`AsyncGraphDatabase.driver`，`get_neo4j_driver()`、`get_neo4j_session()` 用于图库读写；应用启动时 `init_neo4j_driver()`、关闭时 `close_neo4j_driver()`。
