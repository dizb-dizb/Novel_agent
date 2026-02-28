# -*- coding: utf-8 -*-
"""语义检索接口。"""
from typing import List

from .memory import query as _query


def search(query_text: str, book_id: str = None, top_k: int = 5) -> List[dict]:
    """按自然语言问句做语义检索，返回相关片段或知识卡片及来源。"""
    return _query(query_text, book_id=book_id, top_k=top_k)
