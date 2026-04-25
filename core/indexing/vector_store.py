"""向量存储适配器抽象及 Qdrant 实现。"""

from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models

from core.types import ChunkRecord, RetrieverItem


class VectorStoreAdapter:
    """可插拔后端的向量存储接口。"""

    def upsert_text_chunks(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> None:
        """写入文本分块向量。"""
        raise NotImplementedError

    def upsert_parent_docs(self, parents: dict[str, str], vectors: list[list[float]]) -> None:
        """写入父级文档向量。"""
        raise NotImplementedError

    def upsert_image_assets(self, image_records: list[dict[str, Any]], vectors: list[list[float]]) -> None:
        """写入图片资源向量。"""
        raise NotImplementedError

    def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrieverItem]:
        """执行向量检索。"""
        raise NotImplementedError

    def health(self) -> bool:
        """检查后端健康状态。"""
        raise NotImplementedError


class QdrantVectorStoreAdapter(VectorStoreAdapter):
    """用于文本和图片向量检索的 Qdrant 后端。"""

    def __init__(
        self,
        url: str,
        api_key: str,
        text_collection: str,
        image_collection: str,
        parent_collection: str,
        text_dim: int = 512,
        image_dim: int = 512,
        upsert_batch_size: int = 64,
    ):
        """初始化 Qdrant 客户端并确保所需集合存在。"""
        self.client = QdrantClient(url=url, api_key=api_key or None)
        self.text_collection = text_collection
        self.image_collection = image_collection
        self.parent_collection = parent_collection
        self.upsert_batch_size = upsert_batch_size

        self._ensure_collection(self.text_collection, text_dim)
        self._ensure_collection(self.parent_collection, text_dim)
        self._ensure_collection(self.image_collection, image_dim)

    @staticmethod
    def _to_qdrant_id(raw_id: str) -> str:
        """把业务 ID 转成稳定 UUID，满足 Qdrant ID 约束。"""
        return str(uuid5(NAMESPACE_URL, raw_id))

    def _ensure_collection(self, name: str, dim: int) -> None:
        """按需创建指定维度的集合。"""
        exists = self.client.collection_exists(name)
        if exists:
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

    def _upsert_in_batches(
        self,
        collection_name: str,
        points: list[models.PointStruct],
        batch_size: int | None = None,
    ) -> None:
        """分批写入，规避 Qdrant 单请求 payload 32MB 限制。"""
        if not points:
            return

        size = batch_size or self.upsert_batch_size
        idx = 0
        while idx < len(points):
            batch = points[idx : idx + size]
            try:
                self.client.upsert(collection_name=collection_name, points=batch)
                idx += size
            except Exception:
                if size <= 1:
                    raise
                size = max(1, size // 2)

    def upsert_text_chunks(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> None:
        """将文本分块向量写入文本集合。"""
        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            payload = {
                "content": chunk.chunk_text,
                "source": chunk.metadata.get("source_path", ""),
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_id,
                **chunk.metadata,
            }
            qdrant_id = self._to_qdrant_id(chunk.chunk_id)
            points.append(models.PointStruct(id=qdrant_id, vector=vector, payload=payload))

        self._upsert_in_batches(self.text_collection, points)

    def upsert_parent_docs(self, parents: dict[str, str], vectors: list[list[float]]) -> None:
        """将父级文本分块写入补充召回层。"""
        points: list[models.PointStruct] = []
        for (parent_id, text), vector in zip(parents.items(), vectors):
            # Extract doc_id from parent_id (format: "{doc_id}:p:{index}")
            doc_id = parent_id.rsplit(":p:", 1)[0] if ":p:" in parent_id else parent_id
            qdrant_id = self._to_qdrant_id(parent_id)
            points.append(
                models.PointStruct(
                    id=qdrant_id,
                    vector=vector,
                    payload={
                        "content": text,
                        "parent_id": parent_id,
                        "doc_id": doc_id,
                        "source": "parent_doc",
                    },
                )
            )

        self._upsert_in_batches(self.parent_collection, points)

    def delete_text_chunks_by_doc_id(self, doc_id: str) -> None:
        """按 doc_id 删除该文档的所有子块。"""
        self.client.delete(
            collection_name=self.text_collection,
            points_selector=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            ),
        )

    def delete_parent_docs_by_doc_id(self, doc_id: str) -> None:
        """按 doc_id 删除该文档的所有父块。"""
        self.client.delete(
            collection_name=self.parent_collection,
            points_selector=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            ),
        )

    def delete_image_assets_by_doc_id(self, doc_id: str) -> None:
        """按 doc_id 删除该文档的所有图片资产。"""
        self.client.delete(
            collection_name=self.image_collection,
            points_selector=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            ),
        )

    def upsert_image_assets(self, image_records: list[dict[str, Any]], vectors: list[list[float]]) -> None:
        """将图片 embedding 和元数据写入集合。"""
        points: list[models.PointStruct] = []
        for record, vector in zip(image_records, vectors):
            payload = {"content": record.get("caption", ""), "source": record["source_path"], **record}
            qdrant_id = self._to_qdrant_id(record["doc_id"])
            points.append(models.PointStruct(id=qdrant_id, vector=vector, payload=payload))

        self._upsert_in_batches(self.image_collection, points)

    def search_by_ids(self, collection: str, ids: list[str]) -> dict[str, str]:
        """按业务 ID 列表精确查找，返回 {raw_id: content} 映射。

        用于检索后回查父块完整文本。
        """
        if not ids:
            return {}
        qdrant_ids = [self._to_qdrant_id(rid) for rid in ids]
        try:
            points = self.client.retrieve(
                collection_name=collection,
                ids=qdrant_ids,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return {}
        return {
            rid: (pt.payload or {}).get("content", "")
            for rid, pt in zip(ids, points)
            if pt
        }

    def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrieverItem]:
        """按向量检索，并支持可选的元数据精确匹配过滤。"""
        query_filter = None
        if filters:
            conditions = [
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
                for key, value in filters.items()
            ]
            query_filter = models.Filter(must=conditions)

        hits = self.client.search(
            collection_name=collection,
            query_vector=vector,
            limit=top_k,
            query_filter=query_filter,
        )

        items: list[RetrieverItem] = []
        for hit in hits:
            payload = dict(hit.payload)
            logical_id = (
                payload.get("chunk_id")
                or payload.get("parent_id")
                or payload.get("doc_id")
                or str(hit.id)
            )
            items.append(
                RetrieverItem(
                    item_id=str(logical_id),
                    content=payload.get("content", ""),
                    source=payload.get("source", ""),
                    score=float(hit.score),
                    metadata=payload,
                )
            )
        return items

    def health(self) -> bool:
        """返回后端是否可用。"""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
