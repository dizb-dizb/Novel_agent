# -*- coding: utf-8 -*-
"""元协议加载/保存与默认生成。"""
import json
from pathlib import Path
from typing import Optional

from .state_schema import ElementFieldDef, MetaProtocol


def load_protocol(path: Path) -> Optional[MetaProtocol]:
    """从 JSON 文件加载元协议。"""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return MetaProtocol.model_validate(data)
    except Exception:
        return None


def save_protocol(protocol: MetaProtocol, path: Path) -> None:
    """将元协议保存为 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(protocol.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")


def default_protocol(book_id: str) -> MetaProtocol:
    """返回默认元协议（无采样时的兜底）。"""
    return MetaProtocol(
        book_id=book_id,
        element_template=[
            ElementFieldDef(name="名称", kind="str", description="实体或概念名称"),
            ElementFieldDef(name="描述", kind="str", description="简要描述"),
            ElementFieldDef(name="首次出现章节", kind="str", description="chapter_id 或序号"),
        ],
    )
