"""评估数据导入器。

将测试集的全量语料库切块、向量化，导入独立的 Qdrant 评估集合。
评估时从这个集合中检索，与黄金参考文档对比计算检索质量。

核心概念：
- corpus（全量语料库）：所有样本的所有文档去重合并，作为检索源
- golden_docs（黄金参考文档）：每条样本标注的相关文档，用于评估检索是否命中

流程：
1. 从 DatasetLoader 提取全量去重语料库（corpus）
2. 切块 + 向量化
3. 写入独立的 Qdrant 评估集合（collection 名: {dataset}_eval）
4. 评估时检索该集合
5. 评估完成后可选择清理
"""

from __future__ import annotations

import hashlib
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from core.evaluation.dataset_loader import DatasetLoader


class EvalDataIngestor:
    """将测试集文档导入 Qdrant 评估集合。"""

    # 评估集合名称后缀
    EVAL_COLLECTION_SUFFIX = "_eval"

    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str = "",
        embedding_provider: Any = None,
        text_dim: int = 1024,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """初始化评估数据导入器。

        Args:
            qdrant_url: Qdrant 服务地址
            qdrant_api_key: Qdrant API Key
            embedding_provider: EmbeddingProvider 实例，用于向量化
            text_dim: 向量维度
            chunk_size: 文本切块大小（字符数）
            chunk_overlap: 切块重叠大小
        """
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)
        self.embedding_provider = embedding_provider
        self.text_dim = text_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def get_eval_collection_name(self, dataset_name: str) -> str:
        """获取评估集合名称。"""
        return f"{dataset_name}{self.EVAL_COLLECTION_SUFFIX}"

    def ingest_corpus(
        self,
        corpus: list[str],
        dataset_name: str,
    ) -> dict[str, Any]:
        """将全量语料库导入 Qdrant 评估集合。

        Args:
            corpus: 全量去重语料库（由 DatasetLoader.extract_corpus() 提取）
            dataset_name: 数据集名称（用于命名集合）

        Returns:
            导入统计信息
        """
        collection_name = self.get_eval_collection_name(dataset_name)

        # 用哈希去重
        all_docs: dict[str, str] = {}
        for doc_text in corpus:
            if doc_text and len(doc_text.strip()) > 10:
                doc_id = hashlib.md5(doc_text.strip().encode()).hexdigest()[:16]
                all_docs[doc_id] = doc_text.strip()

        print(f"[EvalIngestor] Corpus: {len(corpus)} docs -> {len(all_docs)} unique docs")

        if not all_docs:
            return {"error": "No valid documents to ingest"}

        # 文本切块
        chunks = self._chunk_documents(all_docs)
        print(f"[EvalIngestor] Chunked into {len(chunks)} text chunks")

        # 向量化
        print(f"[EvalIngestor] Embedding {len(chunks)} chunks...")
        chunk_texts = [c["text"] for c in chunks]
        vectors = self._embed_texts(chunk_texts)
        print(f"[EvalIngestor] Got {len(vectors)} vectors")

        # 写入 Qdrant
        self._create_eval_collection(collection_name)
        self._upsert_chunks(collection_name, chunks, vectors)
        print(f"[EvalIngestor] Uploaded to collection: {collection_name}")

        # 验证
        count = self.client.count(collection_name).count
        print(f"[EvalIngestor] Collection {collection_name} now has {count} points")

        return {
            "collection_name": collection_name,
            "total_documents": len(all_docs),
            "total_chunks": len(chunks),
            "vector_dim": len(vectors[0]) if vectors else 0,
        }

    def _chunk_documents(self, docs: dict[str, str]) -> list[dict[str, Any]]:
        """将文档切块。

        Args:
            docs: {doc_id: doc_text}

        Returns:
            切块列表 [{chunk_id, doc_id, text}]
        """
        chunks = []
        for doc_id, text in docs.items():
            doc_chunks = self._split_text(text, self.chunk_size, self.chunk_overlap)
            for i, chunk_text in enumerate(doc_chunks):
                chunks.append({
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "source": f"eval_{doc_id}",
                })
        return chunks

    @staticmethod
    def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        """简单的固定长度切块。"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap
        return chunks

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """向量化文本列表。"""
        if self.embedding_provider is None:
            raise RuntimeError(
                "No embedding provider. "
                "Please ensure EmbeddingProvider is properly initialized."
            )

        all_vectors = []
        batch_size = 32  # 每批处理32条

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # 使用 EmbeddingProvider 的 embed_texts 方法
            try:
                if hasattr(self.embedding_provider, "embed_documents"):
                    vectors = self.embedding_provider.embed_documents(batch)
                elif hasattr(self.embedding_provider, "embed_text"):
                    # 项目的 EmbeddingProvider：逐条调用 embed_text
                    vectors = [self.embedding_provider.embed_text(t) for t in batch]
                elif hasattr(self.embedding_provider, "embed_texts"):
                    vectors = self.embedding_provider.embed_texts(batch)
                elif hasattr(self.embedding_provider, "embed"):
                    vectors = [self.embedding_provider.embed(t) for t in batch]
                else:
                    raise RuntimeError(
                        f"EmbeddingProvider has no compatible method. "
                        f"Available: {[m for m in dir(self.embedding_provider) if not m.startswith('_')]}"
                    )
                all_vectors.extend(vectors)
            except Exception as e:
                print(f"[EvalIngestor] Error embedding batch {i//batch_size}: {e}")
                raise

        return all_vectors

    def _create_eval_collection(self, collection_name: str) -> None:
        """创建评估集合（如果不存在则重建）。"""
        exists = self.client.collection_exists(collection_name)
        if exists:
            print(f"[EvalIngestor] Collection {collection_name} already exists, deleting...")
            self.client.delete_collection(collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.text_dim,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"[EvalIngestor] Created collection: {collection_name}")

    def _upsert_chunks(
        self,
        collection_name: str,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]],
    ) -> None:
        """批量写入切块。"""
        from uuid import NAMESPACE_URL, uuid5

        points = []
        batch_size = 64

        for chunk, vector in zip(chunks, vectors):
            qdrant_id = str(uuid5(NAMESPACE_URL, chunk["chunk_id"]))
            points.append(
                models.PointStruct(
                    id=qdrant_id,
                    vector=vector,
                    payload={
                        "content": chunk["text"],
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"],
                        "source": chunk["source"],
                    },
                )
            )

        # 分批写入
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=collection_name, points=batch)

    def cleanup_collection(self, dataset_name: str) -> bool:
        """清理评估集合。

        Args:
            dataset_name: 数据集名称

        Returns:
            是否成功清理
        """
        collection_name = self.get_eval_collection_name(dataset_name)
        exists = self.client.collection_exists(collection_name)

        if not exists:
            print(f"[EvalIngestor] Collection {collection_name} does not exist, skip cleanup")
            return True

        try:
            self.client.delete_collection(collection_name)
            print(f"[EvalIngestor] Cleaned up collection: {collection_name}")
            return True
        except Exception as e:
            print(f"[EvalIngestor] Failed to cleanup {collection_name}: {e}")
            return False

    def collection_exists(self, dataset_name: str) -> bool:
        """检查评估集合是否已存在。"""
        collection_name = self.get_eval_collection_name(dataset_name)
        return self.client.collection_exists(collection_name)
