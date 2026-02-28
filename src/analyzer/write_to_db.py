# -*- coding: utf-8 -*-
"""
将分析结果（novel_database）写入后端数据库（SQLite Character / Event 表）。
支持：HTTP 调用后端 API，或进程内调用 backend 写入。
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Union

try:
    from src.utils import get_logger
    log = get_logger()
except Exception:
    log = None


def _log(msg: str, *args: Any) -> None:
    if log:
        log.info(msg, *args)


def write_novel_database_to_backend(
    book_id: str,
    novel_database: Union[Path, Dict[str, Any]],
    *,
    backend_url: str = "",
) -> Dict[str, Any]:
    """
    将 novel_database 写入后端数据库（角色表 + 情节树表）。
    :param book_id: 书籍 id
    :param novel_database: novel_database.json 路径或已加载的 dict
    :param backend_url: 后端 API 根地址，如 http://localhost:8000；为空则尝试进程内写入
    :return: {"ok": True, "written": {...}} 或 {"ok": False, "error": "..."}
    """
    if isinstance(novel_database, (Path, str)):
        path = Path(novel_database)
        if not path.is_file():
            return {"ok": False, "error": f"文件不存在: {path}"}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"ok": False, "error": str(e)}
    else:
        data = novel_database

    base = (backend_url or os.getenv("NOVEL_AGENT_BACKEND_URL", "")).strip()
    if base:
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{base.rstrip('/')}/import/analysis",
                data=json.dumps({"book_id": book_id, "novel_database": data}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                out = json.loads(resp.read().decode("utf-8"))
                _log("写入数据库（API）: book_id=%s, written=%s", book_id, out.get("written"))
                return out
        except Exception as e:
            _log("写入数据库 API 失败: %s", e)
            return {"ok": False, "error": str(e)}

    # 进程内写入：依赖 backend 在 path 上
    try:
        async def _run() -> Dict[str, Any]:
            try:
                from backend.database.database import AsyncSessionLocal, init_sqlite_async
                from backend.services.analysis_import_service import write_novel_database_to_db
                import backend.models
            except ImportError:
                from database.database import AsyncSessionLocal, init_sqlite_async
                from services.analysis_import_service import write_novel_database_to_db
                import models
            await init_sqlite_async()
            async with AsyncSessionLocal() as session:
                counts = await write_novel_database_to_db(session, data, book_id)
                await session.commit()
                return {"ok": True, "book_id": book_id, "written": counts}

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # 当前线程没有运行中的 loop，直接 asyncio.run 即可（会创建新 loop）
            return asyncio.run(_run())
        # 已在 async 上下文中，在子线程里跑 asyncio.run 避免冲突
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _run())
            return future.result(timeout=60)
    except Exception as e:
        _log("写入数据库（进程内）失败: %s", e)
        return {"ok": False, "error": str(e)}
