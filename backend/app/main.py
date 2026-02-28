# -*- coding: utf-8 -*-
"""FastAPI 应用入口：网文生成 AI Agent 三大核心数据库交互后端。"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

# 数据库初始化在 lifespan 中执行，避免循环导入时在此处 import database
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from backend.database import init_sqlite_async, init_neo4j_driver, close_neo4j_driver
        import backend.models  # 注册 ORM 表后再建表
    except ImportError:
        from database import init_sqlite_async, init_neo4j_driver, close_neo4j_driver
        import models  # 注册 ORM 表后再建表
    await init_sqlite_async()
    await init_neo4j_driver()
    yield
    await close_neo4j_driver()


app = FastAPI(
    title="NovelAgent API",
    description="网文生成 AI Agent 数据库交互后端（SQLite + Neo4j）",
    lifespan=lifespan,
)

try:
    from backend.routers.import_router import router as import_router
except ImportError:
    from routers.import_router import router as import_router
app.include_router(import_router)


@app.get("/health")
async def health():
    """健康检查。"""
    return {"status": "ok"}
