# -*- coding: utf-8 -*-
"""
数据库配置：SQLAlchemy 异步引擎（SQLite）与 Neo4j 驱动。
- SQLite：用于结构化业务数据（用户、书籍、章节等）。
- Neo4j：用于关系图谱（角色、设定、因果链等）。
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

# ---------- 配置（可从环境变量读取） ----------
import os

# SQLite：默认项目下 data/backend.db，需 aiosqlite 驱动
SQLITE_DIR = Path(__file__).resolve().parents[2] / "data"
SQLITE_PATH = os.getenv("SQLITE_PATH", str(SQLITE_DIR / "backend.db"))
SQLALCHEMY_DATABASE_URI_ASYNC = f"sqlite+aiosqlite:///{SQLITE_PATH}"

# Neo4j：默认本地 bolt
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# ---------- SQLAlchemy 异步引擎与会话 ----------
Base = declarative_base()

# 异步引擎；echo=True 可开启 SQL 日志
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URI_ASYNC,
    echo=os.getenv("SQL_ECHO", "0").lower() in ("1", "true"),
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def init_sqlite_async() -> None:
    """初始化 SQLite：创建 data 目录并创建所有表。"""
    SQLITE_DIR.mkdir(parents=True, exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：获取异步 DB 会话，请求结束后自动关闭。"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------- Neo4j 驱动 ----------
neo4j_driver = None  # 类型: Optional[neo4j.AsyncGraphDatabase.driver]


def get_neo4j_driver():
    """返回已初始化的 Neo4j 异步驱动；若未初始化则返回 None。"""
    return neo4j_driver


async def init_neo4j_driver() -> None:
    """初始化 Neo4j 异步驱动（在 FastAPI lifespan 中调用）。"""
    global neo4j_driver
    try:
        from neo4j import AsyncGraphDatabase
        neo4j_driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        # 验证连接
        await neo4j_driver.verify_connectivity()
    except ImportError:
        neo4j_driver = None  # 未安装 neo4j 时仅禁用图库
    except Exception:
        neo4j_driver = None  # 连接失败时可选：记录日志后置 None 或抛出


async def close_neo4j_driver() -> None:
    """关闭 Neo4j 驱动（在 FastAPI lifespan 中调用）。"""
    global neo4j_driver
    if neo4j_driver is not None:
        await neo4j_driver.close()
        neo4j_driver = None


@asynccontextmanager
async def get_neo4j_session(database: str = "neo4j"):
    """
    获取 Neo4j 异步会话的上下文管理器。
    用法：async with get_neo4j_session() as session: result = await session.run(...)
    """
    if neo4j_driver is None:
        raise RuntimeError("Neo4j 驱动未初始化，请检查 NEO4J_URI 与依赖安装。")
    async with neo4j_driver.session(database=database) as session:
        yield session
