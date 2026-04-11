"""Shared bootstrap helpers for the FastAPI app."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from core.config.settings import AppSettings, load_settings
from core.observability.logging_utils import setup_logging

if TYPE_CHECKING:
    from core.orchestration.pipeline import RAGPipeline


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached application settings."""
    return load_settings()


@lru_cache(maxsize=1)
def get_logger():
    """Return the shared application logger."""
    return setup_logging()


@lru_cache(maxsize=1)
def get_pipeline() -> "RAGPipeline":
    """Bootstrap and cache the RAG pipeline for API requests."""
    from core.indexing.bm25_index import BM25Indexer
    from core.indexing.vector_store import QdrantVectorStoreAdapter
    from core.orchestration.pipeline import RAGPipeline, RetrievalOrchestrator
    from core.providers.model_provider import ModelRegistry
    from core.retrieval.graph_retriever import LightRAGGraphEngine
    from core.retrieval.reranker import build_reranker

    settings = get_settings()
    registry = ModelRegistry(settings.model_registry_path)

    reranker = None
    if settings.rerank_enabled:
        reranker_cfg = registry._cfg.get("reranker", {})
        if reranker_cfg.get("enabled", False):
            reranker = build_reranker(reranker_cfg)

    vector_store = QdrantVectorStoreAdapter(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        text_collection=settings.text_collection,
        image_collection=settings.image_collection,
        parent_collection=settings.parent_collection,
        text_dim=registry.embeddings["text"].get("dimensions", 1024),
        image_dim=registry.embeddings["image"].get("dimensions", 512),
    )
    bm25 = BM25Indexer(settings.bm25_index_file)
    bm25.load()
    graph_engine = LightRAGGraphEngine(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
    )
    orchestrator = RetrievalOrchestrator(
        settings=settings,
        registry=registry,
        vector_store=vector_store,
        bm25=bm25,
        graph_engine=graph_engine,
    )
    return RAGPipeline(settings=settings, orchestrator=orchestrator, registry=registry, reranker=reranker)
