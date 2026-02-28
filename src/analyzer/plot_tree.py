# -*- coding: utf-8 -*-
"""构建与维护剧情树关系。"""
from typing import Dict, List, Optional

from .models import PlotNode


def add_node(tree: Dict[str, PlotNode], node: PlotNode) -> None:
    """将节点加入树。"""
    tree[node.id] = node


def get_children(tree: Dict[str, PlotNode], parent_id: str) -> List[PlotNode]:
    """获取某节点的子节点列表。"""
    return [n for n in tree.values() if n.parent_id == parent_id]
