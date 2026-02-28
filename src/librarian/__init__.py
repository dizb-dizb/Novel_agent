# -*- coding: utf-8 -*-
"""模块3：图书管理 AI（索引 + 风格样本 + 改写上下文）。"""
from .memory import get_collection, add_documents, query
from .manager import get_book_index, generate_style_report
from .search_engine import search
from .style_store import (
    StyleStore,
    StyleSample,
    StyleFingerprint,
    build_style_fingerprint_library,
)
from .context_loader import build_rewrite_context

__all__ = [
    "get_collection",
    "add_documents",
    "query",
    "get_book_index",
    "generate_style_report",
    "search",
    "StyleStore",
    "StyleSample",
    "StyleFingerprint",
    "build_style_fingerprint_library",
    "build_rewrite_context",
]
