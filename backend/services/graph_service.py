# -*- coding: utf-8 -*-
"""
Neo4j 关系图谱操作封装：角色节点与关系边的创建/更新、关系网查询。
关系边属性：relation_type（如仇人、师徒）、affinity_score（好感度）、event_id（关联事件 ID）。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)


class RelationshipEdge(TypedDict, total=False):
    """关系边属性（用于返回）。"""
    relation_type: str
    affinity_score: float
    event_id: Optional[str]


class CharacterNodeInNetwork(TypedDict):
    """关系网中的角色节点（一度邻居）。"""
    char_id: str
    relation_type: str
    affinity_score: float
    event_id: Optional[str]
    direction: str  # "outgoing" | "incoming"


class GraphServiceError(Exception):
    """图服务异常（如驱动未就绪、Neo4j 执行失败）。"""
    pass


def _get_driver():
    """获取 Neo4j 驱动；兼容从 backend 或项目根运行。"""
    try:
        from backend.database.database import get_neo4j_driver
        return get_neo4j_driver()
    except ImportError:
        from database.database import get_neo4j_driver
        return get_neo4j_driver()


async def create_or_update_relationship(
    char_a_id: str,
    char_b_id: str,
    relation_type: str,
    affinity_score: float,
    event_id: Optional[str] = None,
    database: str = "neo4j",
) -> Dict[str, Any]:
    """
    创建或更新两个角色节点之间的关系边。
    若节点不存在则先 MERGE 创建，再 MERGE 关系并设置属性。

    :param char_a_id: 角色 A 的 ID（与角色库 Character.id 对应）
    :param char_b_id: 角色 B 的 ID
    :param relation_type: 关系类型，如 "仇人"、"师徒"、"师徒"
    :param affinity_score: 好感度数值（可负值表示敌对）
    :param event_id: 触发该关系的关联事件/情节节点 ID，可选
    :param database: Neo4j 数据库名
    :return: 包含 created/updated 等信息的摘要
    :raises GraphServiceError: 驱动未就绪或 Neo4j 执行失败
    """
    driver = _get_driver()
    if driver is None:
        raise GraphServiceError("Neo4j 驱动未初始化，请检查 NEO4J_URI 与依赖。")

    async def _tx(tx: Any) -> Dict[str, Any]:
        # 有向边：A -> B；若需双向可再建 B -> A 或改用无向语义
        result = await tx.run(
            """
            MERGE (a:Character {id: $char_a_id})
            MERGE (b:Character {id: $char_b_id})
            WITH a, b
            MERGE (a)-[r:RELATES_TO]->(b)
            SET r.relation_type = $relation_type,
                r.affinity_score = $affinity_score,
                r.event_id = $event_id
            RETURN id(r) AS rel_id
            """,
            char_a_id=char_a_id,
            char_b_id=char_b_id,
            relation_type=relation_type,
            affinity_score=affinity_score,
            event_id=event_id or "",
        )
        # 消费 result 并取首条（neo4j 5.x 异步 API 兼容）
        record = None
        async for rec in result:
            record = rec
            break
        return {"rel_id": record["rel_id"] if record else None, "updated": True}

    try:
        async with driver.session(database=database) as session:
            summary = await session.execute_write(_tx)
            return {"ok": True, **summary}
    except Exception as e:
        logger.exception("create_or_update_relationship failed: %s", e)
        raise GraphServiceError(f"Neo4j 执行失败: {e}") from e


async def get_character_network(
    char_id: str,
    depth: int = 1,
    database: str = "neo4j",
) -> List[CharacterNodeInNetwork]:
    """
    查询某角色的关系网，返回指定深度内的邻居节点及关系属性。
    depth=1 表示一度关系（直接相连的角色及关系类型、好感度、event_id）。

    :param char_id: 角色 ID
    :param depth: 遍历深度，默认 1（一度邻居）
    :param database: Neo4j 数据库名
    :return: 列表，每项包含邻居 char_id、relation_type、affinity_score、event_id、direction
    :raises GraphServiceError: 驱动未就绪或 Neo4j 执行失败
    """
    driver = _get_driver()
    if driver is None:
        raise GraphServiceError("Neo4j 驱动未初始化，请检查 NEO4J_URI 与依赖。")

    if depth < 1:
        depth = 1

    out: List[CharacterNodeInNetwork] = []
    try:
        async with driver.session(database=database) as session:
            if depth == 1:
                # 一度：直接 (c)-[r:RELATES_TO]-(other)，区分 outgoing / incoming
                result = await session.run(
                    """
                    MATCH (c:Character {id: $char_id})-[r:RELATES_TO]-(other:Character)
                    RETURN other.id AS other_id,
                           r.relation_type AS relation_type,
                           r.affinity_score AS affinity_score,
                           r.event_id AS event_id,
                           CASE WHEN startNode(r) = c THEN 'outgoing' ELSE 'incoming' END AS direction
                    """,
                    char_id=char_id,
                )
            else:
                # 多度：可变长度路径，每个邻居取一条路径的最后一跳关系属性
                result = await session.run(
                    """
                    MATCH path = (c:Character {id: $char_id})-[:RELATES_TO*1..$depth]-(other:Character)
                    WHERE c <> other
                    WITH other, relationships(path) AS rels
                    WITH other, rels[size(rels)-1] AS last_r
                    RETURN DISTINCT other.id AS other_id,
                           last_r.relation_type AS relation_type,
                           last_r.affinity_score AS affinity_score,
                           last_r.event_id AS event_id,
                           'outgoing' AS direction
                    """,
                    char_id=char_id,
                    depth=depth,
                )
            async for record in result:
                out.append(CharacterNodeInNetwork(
                    char_id=record["other_id"] or "",
                    relation_type=record["relation_type"] or "",
                    affinity_score=float(record["affinity_score"] or 0),
                    event_id=record["event_id"] or None,
                    direction=record["direction"] or "outgoing",
                ))
    except Exception as e:
        logger.exception("get_character_network failed: %s", e)
        raise GraphServiceError(f"Neo4j 执行失败: {e}") from e

    return out
