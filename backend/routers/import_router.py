# -*- coding: utf-8 -*-
"""分析结果写入数据库：导入 novel_database 到 SQLite。"""
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException

try:
    from backend.database import get_async_session
    from backend.services.analysis_import_service import write_novel_database_to_db
except ImportError:
    from database import get_async_session
    from services.analysis_import_service import write_novel_database_to_db

from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/import", tags=["import"])


@router.post("/analysis")
async def import_analysis(
    body: Dict[str, Any],
    session: AsyncSession = Depends(get_async_session),
) -> Dict[str, Any]:
    """
    将分析结果写入数据库。
    Body: { "book_id": "书籍id", "novel_database": { ... } }
    novel_database 为 novel_database.json 的完整对象（含 book_id, title, entities_by_type, plot_tree, cards 等）。
    """
    book_id = (body.get("book_id") or "").strip()
    novel_db = body.get("novel_database")
    if not novel_db or not isinstance(novel_db, dict):
        raise HTTPException(status_code=400, detail="需要 novel_database 对象")
    if not book_id:
        book_id = (novel_db.get("book_id") or "").strip()
    try:
        counts = await write_novel_database_to_db(session, novel_db, book_id)
        return {"ok": True, "book_id": book_id, "written": counts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
