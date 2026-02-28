# -*- coding: utf-8 -*-
"""
提取缓存：按 (book_id, chapter_id, content_hash, protocol_sig) 缓存单章提取结果。
同一章节、同一模板多次提取时命中缓存，避免重复调用低质量模型（用于「命中缓存四次提取优化」）。
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

from .models import KnowledgeCard, PlotNode
from .state_schema import MetaProtocol


def _content_hash(content: str) -> str:
    """章节正文的短哈希，用于缓存键。"""
    return hashlib.md5((content or "").encode("utf-8")).hexdigest()[:16]


def _protocol_sig(protocol: Optional[MetaProtocol]) -> str:
    """元协议签名，模板变化则缓存失效。"""
    if not protocol:
        return "no_protocol"
    return hashlib.md5(protocol.model_dump_json().encode("utf-8")).hexdigest()[:16]


def _cache_key(book_id: str, chapter_id: str, content_hash: str, protocol_sig: str) -> Tuple[str, str, str, str]:
    return (book_id or "", chapter_id or "", content_hash, protocol_sig)


class ExtractionCache:
    """
    内存缓存：key=(book_id, chapter_id, content_hash, protocol_sig), value=(cards, nodes)。
    同一本书、同一章、同一模板版本只提取一次，后续命中缓存。
    """

    def __init__(self, max_size: int = 10000) -> None:
        self._store: Dict[Tuple[str, str, str, str], Tuple[List[Dict], List[Dict]]] = {}
        self._max_size = max_size
        self._order: List[Tuple[str, str, str, str]] = []

    def get(
        self,
        book_id: str,
        chapter_id: str,
        content: str,
        protocol: Optional[MetaProtocol],
    ) -> Optional[Tuple[List[KnowledgeCard], List[PlotNode]]]:
        """若存在则返回 (cards, nodes)，否则返回 None。"""
        key = _cache_key(book_id, chapter_id, _content_hash(content), _protocol_sig(protocol))
        raw = self._store.get(key)
        if raw is None:
            return None
        cards_data, nodes_data = raw
        cards = [KnowledgeCard.model_validate(c) for c in cards_data]
        nodes = [PlotNode.model_validate(n) for n in nodes_data]
        return cards, nodes

    def set(
        self,
        book_id: str,
        chapter_id: str,
        content: str,
        protocol: Optional[MetaProtocol],
        cards: List[KnowledgeCard],
        nodes: List[PlotNode],
    ) -> None:
        """写入缓存。"""
        key = _cache_key(book_id, chapter_id, _content_hash(content), _protocol_sig(protocol))
        cards_data = [c.model_dump() for c in cards]
        nodes_data = [n.model_dump() for n in nodes]
        if key not in self._store and len(self._store) >= self._max_size and self._order:
            old = self._order.pop(0)
            self._store.pop(old, None)
        self._store[key] = (cards_data, nodes_data)
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def clear(self) -> None:
        """清空缓存。"""
        self._store.clear()
        self._order.clear()


# 全局单例，供 pipeline 使用；可按 book 或进程复用
_global_cache: Optional[ExtractionCache] = None


def get_extraction_cache() -> ExtractionCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = ExtractionCache()
    return _global_cache
