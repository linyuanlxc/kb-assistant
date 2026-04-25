"""LabKB 的索引构建流水线。

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
from typing import Any

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
    """构建并更新所有检索索引。"""

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
            text_dim=self.registry.embeddings["text"].get("dimensions", 512),
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

    def _detect_deleted(
        self,
        manifest: IndexManifest,
        current_text_ids: set[str],
        current_image_ids: set[str],
    ) -> tuple[set[str], set[str]]:
        """检测已从源目录删除的文件。

        Returns:
            deleted_text_ids: 已删除的文本文档 ID 集合
            deleted_image_ids: 已删除的图片文档 ID 集合
        """
        deleted_text = set()
        for doc_id in list(manifest.docs.keys()):
            if doc_id not in current_text_ids:
                deleted_text.add(doc_id)

        deleted_image = set()
        for doc_id in list(manifest.assets.keys()):
            if doc_id not in current_image_ids:
                deleted_image.add(doc_id)

        return deleted_text, deleted_image

    def _detect_moved(
        self,
        deleted_text_ids: set[str],
        deleted_image_ids: set[str],
        new_text_docs: list[RawDocument],
        new_image_docs: list[RawDocument],
        manifest: IndexManifest,
    ) -> tuple[list[RawDocument], list[RawDocument]]:
        """基于校验和匹配检测文件移动/重命名。

        对于 manifest 中已删除的文件，如果新文件中有相同 checksum 的，
        则认为是移动/重命名，复用原 doc_id，并更新 source_path。
        返回更新后的 new_text_docs 和 new_image_docs（已复用旧 doc_id）。
        """
        # 构建已删除文件的 checksum -> doc_id 映射
        deleted_text_checksums: dict[str, str] = {}
        for doc_id in deleted_text_ids:
            entry = manifest.docs.get(doc_id)
            if entry:
                deleted_text_checksums[entry["checksum"]] = doc_id

        deleted_image_checksums: dict[str, str] = {}
        for doc_id in deleted_image_ids:
            entry = manifest.assets.get(doc_id)
            if entry:
                deleted_image_checksums[entry["checksum"]] = doc_id

        # 检查新文件是否与已删除文件有相同 checksum
        remaining_new_text: list[RawDocument] = []
        remaining_new_image: list[RawDocument] = []

        for d in new_text_docs:
            matched_old_id = deleted_text_checksums.get(d.checksum)
            if matched_old_id:
                # 找到匹配的旧文件，认为是移动/重命名
                self.logger.info(
                    "moved/renamed text doc detected",
                    extra={"old_doc_id": matched_old_id, "new_path": d.source_path},
                )
                # 从 deleted 中移除（不再是删除，而是移动）
                deleted_text_ids.discard(matched_old_id)
                # 复用旧 doc_id
                d.doc_id = matched_old_id
                # 更新 manifest 中的 source_path
                manifest.docs[matched_old_id]["source_path"] = d.source_path
                manifest.docs[matched_old_id]["updated_at"] = datetime.now().isoformat()
                remaining_new_text.append(d)
            else:
                remaining_new_text.append(d)

        for d in new_image_docs:
            matched_old_id = deleted_image_checksums.get(d.checksum)
            if matched_old_id:
                self.logger.info(
                    "moved/renamed image detected",
                    extra={"old_doc_id": matched_old_id, "new_path": d.source_path},
                )
                deleted_image_ids.discard(matched_old_id)
                d.doc_id = matched_old_id
                manifest.assets[matched_old_id]["source_path"] = d.source_path
                manifest.assets[matched_old_id]["updated_at"] = datetime.now().isoformat()
                remaining_new_image.append(d)
            else:
                remaining_new_image.append(d)

        return remaining_new_text, remaining_new_image

    def _cleanup_deleted(
        self,
        deleted_text_ids: set[str],
        deleted_image_ids: set[str],
        manifest: IndexManifest,
    ) -> None:
        """清理已删除文件的所有索引数据。"""
        # 清理文本文档
        for doc_id in deleted_text_ids:
            self.logger.info("cleaning up deleted text doc", extra={"doc_id": doc_id})
            # 从 Qdrant 删除
            self.vector_store.delete_text_chunks_by_doc_id(doc_id)
            self.vector_store.delete_parent_docs_by_doc_id(doc_id)
            # 从 BM25 records 中删除并重建
            if self.bm25.records:
                changed_doc_ids = {doc_id}
                self.bm25.build([], doc_ids=changed_doc_ids)
                self.bm25.save()
            # 从 Neo4j 删除
            if self.graph_engine.health():
                self.graph_engine.delete_doc(doc_id)
            # 从 manifest 删除
            manifest.docs.pop(doc_id, None)

        # 清理图片
        for doc_id in deleted_image_ids:
            self.logger.info("cleaning up deleted image", extra={"doc_id": doc_id})
            self.vector_store.delete_image_assets_by_doc_id(doc_id)
            manifest.assets.pop(doc_id, None)

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
        self.logger.info("index build started", extra={"incremental": incremental})
        manifest = load_manifest(self.settings.manifest_path)

        # 加载文档（现在返回 3 个值）
        text_docs, image_docs, structured_docs = load_raw_documents(self.settings.source_dir)
        self.logger.info(
            "source loaded",
            extra={"text_docs": len(text_docs), "image_docs": len(image_docs)},
        )

        # 当前文件 ID 集合（用于检测删除）
        current_text_ids = {d.doc_id for d in text_docs}
        current_image_ids = {d.doc_id for d in image_docs}

        # 检测已删除的文件
        deleted_text_ids, deleted_image_ids = self._detect_deleted(
            manifest, current_text_ids, current_image_ids
        )
        self.logger.info(
            "deleted docs detected",
            extra={"deleted_text": len(deleted_text_ids), "deleted_image": len(deleted_image_ids)},
        )

        # 找出新文件（manifest 中无记录）
        existing_text_ids = {d.doc_id for d in text_docs if d.doc_id in manifest.docs}
        existing_image_ids = {d.doc_id for d in image_docs if d.doc_id in manifest.assets}
        new_text = [d for d in text_docs if d.doc_id not in existing_text_ids]
        new_image = [d for d in image_docs if d.doc_id not in existing_image_ids]

        # 基于校验和匹配检测移动/重命名
        new_text, new_image = self._detect_moved(
            deleted_text_ids, deleted_image_ids,
            new_text, new_image,
            manifest,
        )

        # 合并：existing + new（已处理移动检测）= 所有当前文件
        # 但 incremental_filter 需要接收所有当前文件
        # 所以这里不需要合并，直接用原始的 text_docs/image_docs 即可

        # 先清理已删除的文件
        if deleted_text_ids or deleted_image_ids:
            self._cleanup_deleted(deleted_text_ids, deleted_image_ids, manifest)
            # 删除后需要重新加载 BM25 records（cleanup 中已经处理了）
            # 注意：如果 BM25 是增量模式，build() 会自己过滤

        # 增量过滤
        if incremental:
            text_docs, image_docs = self._incremental_filter(text_docs, image_docs, manifest)
            self.logger.info(
                "incremental filter result",
                extra={"changed_text": len(text_docs), "changed_image": len(image_docs)},
            )

        # 切分父子块
        chunks: list[ChunkRecord] = []
        parent_map: dict[str, str] = {}
        if text_docs:
            chunks, parent_map = split_parent_child(text_docs, structured_docs=structured_docs)

        # 对变更的文本文档，先删除旧数据再写入
        changed_text_ids = {d.doc_id for d in text_docs}
        changed_image_ids = {d.doc_id for d in image_docs}

        if chunks:
            # 先删除旧子块和旧父块
            for d in text_docs:
                self.vector_store.delete_text_chunks_by_doc_id(d.doc_id)
                self.vector_store.delete_parent_docs_by_doc_id(d.doc_id)
                if self.graph_engine.health():
                    self.graph_engine.delete_doc(d.doc_id)
            # BM25 增量合并：传入变更的 doc_ids
            bm25_doc_ids = {d.doc_id for d in text_docs}
            # 先加载现有 BM25 records
            self.bm25.load()
            # 构建（增量模式）
            self.bm25.build(chunks, doc_ids=bm25_doc_ids)
            self.bm25.save()

            # 写入新向量
            child_vectors = self.embedder.embed_text([c.chunk_text for c in chunks])
            self.vector_store.upsert_text_chunks(chunks, child_vectors)

            parent_ids = list(parent_map.keys())
            parent_vectors = self.embedder.embed_text([parent_map[k] for k in parent_ids])
            self.vector_store.upsert_parent_docs(parent_map, parent_vectors)

            # 更新图谱
            if parent_map and self.graph_engine.health():
                self.graph_engine.upsert_from_parent_docs(parent_map)

        # 构建图片向量
        if image_docs:
            for d in image_docs:
                self.vector_store.delete_image_assets_by_doc_id(d.doc_id)
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

        # 更新 manifest
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
            "deleted_text_docs": len(deleted_text_ids),
            "deleted_image_docs": len(deleted_image_ids),
            "graph_enabled": self.graph_engine.health(),
        }
        self.logger.info("index build finished", extra={"extra_data": summary})
        return summary
