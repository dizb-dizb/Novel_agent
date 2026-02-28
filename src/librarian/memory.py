# -*- coding: utf-8 -*-
"""向量数据库封装（RAG），持久化路径为 data/db。"""
from pathlib import Path
from typing import List, Optional

# 默认使用项目根下 data/db
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "db"


def get_collection(name: str = "novel_cards", persist_directory: Optional[Path] = None):
    """获取或创建 Chroma 集合（占位：需安装 chromadb 后实现）。"""
    # TODO: import chromadb; persist_directory = persist_directory or DEFAULT_DB_PATH
    return None


def add_documents(book_id: str, texts: List[str], metadatas: List[dict] = None) -> None:
    """写入文档到向量库（占位）。"""
    pass


def query(question: str, book_id: str = None, top_k: int = 5) -> List[dict]:
    """语义检索（占位）。"""
    return []
