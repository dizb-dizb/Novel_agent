# -*- coding: utf-8 -*-
"""
整本仿写用 Pydantic 协议（State & Style）—— 实现在 schemas，此处仅 re-export。
与 SQLAlchemy ORM 的 models 区分：此为各 Agent 间传递用的数据结构。
"""
try:
    from backend.schemas.orchestrator_models import BookState, ChapterDesign, StyleGuide
except ImportError:
    from schemas.orchestrator_models import BookState, ChapterDesign, StyleGuide

__all__ = ["BookState", "ChapterDesign", "StyleGuide"]
