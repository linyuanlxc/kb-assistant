"""基于 Neo4j 的图检索器，用于 LightRAG 风格的证据召回。

当前实现以关键词实体提取作为务实基线，
再在 Neo4j 中把文档和实体连接起来完成图检索。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import jieba.analyse
from neo4j import GraphDatabase

from core.types import RetrieverItem


class LightRAGGraphEngine:
    """基于 Neo4j 的图检索引擎。"""

    def __init__(self, neo4j_uri: str, user: str, password: str):
        """初始化 Neo4j 驱动。"""
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))

    def close(self) -> None:
        """关闭 Neo4j 连接。"""
        self.driver.close()

    def health(self) -> bool:
        """检查 Neo4j 端点是否可达。"""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception:
            return False

    def delete_doc(self, doc_id: str) -> None:
        """删除文档节点及其所有关系，并清理不再被引用的实体。"""
        with self.driver.session() as session:
            # 先获取该文档关联的所有实体
            result = session.run(
                "MATCH (d:Doc {id:$pid})-[:MENTIONS]->(e:Entity) RETURN e.name AS name",
                pid=doc_id,
            )
            entity_names = [record["name"] for record in result]

            # 删除文档节点及其所有关系
            session.run(
                "MATCH (d:Doc {id:$pid}) DETACH DELETE d",
                pid=doc_id,
            )

            # 清理不再被任何文档引用的实体
            for name in entity_names:
                session.run(
                    "MATCH (e:Entity {name:$name}) "
                    "OPTIONAL MATCH (e)<-[:MENTIONS]-(d:Doc) "
                    "WITH e, d "
                    "WHERE d IS NULL "
                    "DELETE e",
                    name=name,
                )

    @staticmethod
    def _extract_entities(text: str, topk: int = 8) -> list[str]:
        """从文本中提取类实体关键词。"""
        return [w for w in jieba.analyse.extract_tags(text, topK=topk) if len(w) >= 2]

    def upsert_from_parent_docs(self, parents: dict[str, str]) -> None:
        """将父级文档及其实体关系写入图谱。"""
        with self.driver.session() as session:
            for pid, text in parents.items():
                entities = self._extract_entities(text)
                session.run(
                    "MERGE (d:Doc {id:$pid}) SET d.content = $content",
                    pid=pid,
                    content=text[:4000],
                )

                # 建立文档到实体的关联边。
                for entity in entities:
                    session.run(
                        "MERGE (n:Entity {name:$name}) WITH n MATCH (d:Doc {id:$pid}) MERGE (d)-[:MENTIONS]->(n)",
                        name=entity,
                        pid=pid,
                    )

                # 构建实体之间的共现关系图。
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        session.run(
                            "MATCH (a:Entity {name:$a}), (b:Entity {name:$b}) MERGE (a)-[r:RELATED]->(b) ON CREATE SET r.weight=1 ON MATCH SET r.weight=r.weight+1",
                            a=entities[i],
                            b=entities[j],
                        )

    def search(self, query: str, top_k: int = 8) -> tuple[list[RetrieverItem], list[dict[str, Any]]]:
        """根据查询实体检索文档，并返回证据链接。"""
        entities = self._extract_entities(query, topk=5)
        if not entities:
            return [], []

        doc_scores = defaultdict(float)
        evidence: list[dict[str, Any]] = []
        with self.driver.session() as session:
            for entity in entities:
                rows = session.run(
                    "MATCH (n:Entity {name:$name})<-[:MENTIONS]-(d:Doc) RETURN d.id AS doc_id, d.content AS content LIMIT 20",
                    name=entity,
                )
                for row in rows:
                    doc_scores[row["doc_id"]] += 1.0
                    evidence.append({"entity": entity, "doc_id": row["doc_id"]})

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        items = [
            RetrieverItem(
                item_id=doc_id,
                content="",
                source="neo4j_graph",
                score=float(score),
                metadata={"doc_id": doc_id},
            )
            for doc_id, score in ranked
        ]
        return items, evidence
