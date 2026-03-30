"""KB Assistant V2 的索引构建流水线。

本模块负责全量与增量索引流程：
1. 从源目录加载文本和图片文档。
2. 将文本文档切分为父子分块。
3. 构建或更新 Qdrant 向量集合。
4. 构建或更新 BM25 索引。
5. 构建或更新 Neo4j 中的 LightRAG 图谱。
6. 持久化索引清单和版本记录。
"""

from __future__ import annotations

from datetime import datetime

from core.config.settings import AppSettings
from core.indexing.bm25_index import BM25Indexer
from core.indexing.manifest import load_manifest, register_version, save_manifest
from core.indexing.vector_store import QdrantVectorStoreAdapter
from core.ingestion.loaders import load_raw_documents
from core.ingestion.splitter import split_parent_child
from core.observability.logging_utils import setup_logging
from core.providers.model_provider import ModelRegistry
from core.retrieval.graph_retriever import LightRAGGraphEngine
from core.types import ChunkRecord, IndexManifest, RawDocument


class IndexBuilder:
    """构建并更新所有检索索引。

    实现上优先保证稳定性和可观测性：
    - 记录清晰的步骤级日志
    - 通过源路径生成确定性的 ID
    - 基于校验和更新 manifest
    """

    def __init__(self, settings: AppSettings):
        """初始化索引构建器及其依赖组件。"""
        self.settings = settings
        self.logger = setup_logging()
        self.registry = ModelRegistry(settings.model_registry_path)
        self.embedder = self.registry.build_embedding()
        self.vector_store = QdrantVectorStoreAdapter(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            text_collection=settings.text_collection,
            image_collection=settings.image_collection,
            parent_collection=settings.parent_collection,
            text_dim=self.registry.embeddings["text"].get("dimensions", 1024),
            image_dim=self.registry.embeddings["image"].get("dimensions", 512),
        )
        self.bm25 = BM25Indexer(settings.bm25_index_file)
        self.graph_engine = LightRAGGraphEngine(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
        )

    def _incremental_filter(
        self,
        text_docs: list[RawDocument],
        image_docs: list[RawDocument],
        manifest: IndexManifest,
    ) -> tuple[list[RawDocument], list[RawDocument]]:
        """通过校验和比较，仅保留新增或变更文档。"""
        changed_text = []
        for d in text_docs:
            prev = manifest.docs.get(d.doc_id)
            if not prev or prev.get("checksum") != d.checksum:
                changed_text.append(d)

        changed_image = []
        for d in image_docs:
            prev = manifest.assets.get(d.doc_id)
            if not prev or prev.get("checksum") != d.checksum:
                changed_image.append(d)

        return changed_text, changed_image

    def _update_manifest(
        self,
        manifest: IndexManifest,
        text_docs: list[RawDocument],
        image_docs: list[RawDocument],
    ) -> None:
        """将本次处理到的文档信息写回 manifest。"""
        for d in text_docs:
            manifest.docs[d.doc_id] = {
                "checksum": d.checksum,
                "source_path": d.source_path,
                "updated_at": datetime.now().isoformat(),
            }
        for d in image_docs:
            manifest.assets[d.doc_id] = {
                "checksum": d.checksum,
                "source_path": d.source_path,
                "updated_at": datetime.now().isoformat(),
            }

    def build(self, incremental: bool = True) -> dict:
        """执行完整的索引构建流程。

        Args:
            incremental: 为 True 时仅处理新增或变更文件。
        """
        self.logger.info("index build started")
        manifest = load_manifest(self.settings.manifest_path)

        text_docs, image_docs = load_raw_documents(self.settings.source_dir)
        self.logger.info(
            "source loaded",
            extra={"extra_data": {"text_docs": len(text_docs), "image_docs": len(image_docs)}},
        )

        if incremental:
            text_docs, image_docs = self._incremental_filter(text_docs, image_docs, manifest)

        chunks: list[ChunkRecord] = []
        parent_map: dict[str, str] = {}
        if text_docs:
            chunks, parent_map = split_parent_child(text_docs)

        # 1）构建文本分块向量。
        if chunks:
            child_vectors = self.embedder.embed_text([c.chunk_text for c in chunks])
            self.vector_store.upsert_text_chunks(chunks, child_vectors)

            parent_ids = list(parent_map.keys())
            parent_vectors = self.embedder.embed_text([parent_map[k] for k in parent_ids])
            self.vector_store.upsert_parent_docs(parent_map, parent_vectors)

            self.bm25.build(chunks)
            self.bm25.save()

        # 2）构建图片向量。
        if image_docs:
            records = [
                {
                    "doc_id": d.doc_id,
                    "source_path": d.source_path,
                    "caption": d.title or "",
                    "modality": "image",
                    "checksum": d.checksum,
                }
                for d in image_docs
            ]
            vectors = self.embedder.embed_image_paths([d.source_path for d in image_docs])
            self.vector_store.upsert_image_assets(records, vectors)

        # 3）更新图谱索引。
        if parent_map and self.graph_engine.health():
            self.graph_engine.upsert_from_parent_docs(parent_map)

        self._update_manifest(manifest, text_docs, image_docs)
        register_version(
            manifest,
            note=f"incremental={incremental}, chunks={len(chunks)}, image_docs={len(image_docs)}",
        )
        save_manifest(self.settings.manifest_path, manifest)

        summary = {
            "text_docs_processed": len(text_docs),
            "image_docs_processed": len(image_docs),
            "chunks_processed": len(chunks),
            "graph_enabled": self.graph_engine.health(),
        }
        self.logger.info("index build finished", extra={"extra_data": summary})
        return summary
