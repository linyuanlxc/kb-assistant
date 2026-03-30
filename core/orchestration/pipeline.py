"""RAG 编排模块。

职责：
1. 协调多路检索（Dense/BM25/Graph/Image）。
2. 执行融合排序与调试信息输出。
3. 构建上下文并调用大模型流式回答。
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from core.config.settings import AppSettings
from core.generation.prompting import build_answer_messages, build_context, build_rewrite_messages
from core.indexing.bm25_index import BM25Indexer
from core.indexing.vector_store import QdrantVectorStoreAdapter
from core.providers.model_provider import ModelRegistry
from core.retrieval.fusion import normalize_weights, reciprocal_rank_fusion, weighted_merge
from core.retrieval.graph_retriever import LightRAGGraphEngine
from core.types import RetrieverRequest, RetrieverResult


class RetrievalOrchestrator:
    """检索编排器：负责多路召回与融合。"""

    def __init__(
        self,
        settings: AppSettings,
        registry: ModelRegistry,
        vector_store: QdrantVectorStoreAdapter,
        bm25: BM25Indexer,
        graph_engine: LightRAGGraphEngine,
    ):
        self.settings = settings
        self.registry = registry
        self.vector_store = vector_store
        self.bm25 = bm25
        self.graph_engine = graph_engine
        self.embedder = registry.build_embedding()

    def retrieve(self, req: RetrieverRequest) -> RetrieverResult:
        """执行检索并返回融合结果。"""
        start = time.perf_counter()
        query_vec = self.embedder.embed_query(req.query)

        dense_items = self.vector_store.search(
            collection=self.settings.text_collection,
            vector=query_vec,
            top_k=max(req.top_k * 3, 20),
            filters=req.filters,
        )
        bm25_items = self.bm25.search(req.query, top_k=max(req.top_k * 3, 20))

        graph_items: list[Any] = []
        graph_evidence: list[dict[str, Any]] = []
        if self.graph_engine.health():
            graph_items, graph_evidence = self.graph_engine.search(
                req.query,
                top_k=max(req.top_k * 2, 12),
            )

        image_items: list[Any] = []
        if req.image_inputs:
            image_vectors = self.embedder.embed_image_paths(req.image_inputs)
            if image_vectors:
                image_items = self.vector_store.search(
                    collection=self.settings.image_collection,
                    vector=image_vectors[0],
                    top_k=max(req.top_k * 2, 10),
                )

        # 第一层：按业务权重融合。
        weights = normalize_weights(self.settings.retrieval_weights, use_image=bool(req.image_inputs))
        merged_items, merged_scores = weighted_merge(
            dense_items,
            bm25_items,
            graph_items,
            image_items,
            weights,
            req.top_k,
        )

        # 第二层：使用 RRF 进行名次补偿。
        rrf_scores = reciprocal_rank_fusion([dense_items, bm25_items, graph_items, image_items])
        for item in merged_items:
            item.score += rrf_scores.get(item.item_id, 0.0)
        merged_items = sorted(merged_items, key=lambda item: item.score, reverse=True)

        debug_info: dict[str, Any] = {}
        if req.debug:
            debug_info = {
                "dense_top": [asdict(item) for item in dense_items[:5]],
                "bm25_top": [asdict(item) for item in bm25_items[:5]],
                "graph_top": [asdict(item) for item in graph_items[:5]],
                "image_top": [asdict(item) for item in image_items[:5]],
                "weights": weights,
                "rrf_scores": rrf_scores,
            }

        latency_ms = (time.perf_counter() - start) * 1000
        return RetrieverResult(
            items=merged_items,
            scores=merged_scores,
            sources=sorted({item.source for item in merged_items if item.source}),
            graph_evidence=graph_evidence,
            latency_ms=round(latency_ms, 2),
            debug_info=debug_info,
        )


class RAGPipeline:
    """端到端问答流水线。"""

    def __init__(self, settings: AppSettings, orchestrator: RetrievalOrchestrator, registry: ModelRegistry):
        self.settings = settings
        self.orchestrator = orchestrator
        self.fast_model = registry.build_chat("fast_model")
        self.quality_model = registry.build_chat("quality_model")

    def rewrite_query(self, query: str, chat_history: list[tuple[str, str]]) -> str:
        """在多轮对话场景下做查询改写。"""
        if not chat_history:
            return query
        messages = build_rewrite_messages(query, chat_history)
        rewritten = self.fast_model.generate(messages, temperature=0.1)
        return rewritten.strip() or query

    def answer_stream(self, req: RetrieverRequest):
        """返回流式回答生成器，并附带检索元信息。"""
        rewritten_query = self.rewrite_query(req.query, req.chat_history)

        retrieval_req = RetrieverRequest(
            query=rewritten_query,
            chat_history=req.chat_history,
            filters=req.filters,
            modality=req.modality,
            image_inputs=req.image_inputs,
            top_k=req.top_k,
            debug=req.debug,
        )
        retrieval_result = self.orchestrator.retrieve(retrieval_req)

        context = build_context([asdict(item) for item in retrieval_result.items])
        messages = build_answer_messages(req.query, context, req.chat_history)

        stream = self.quality_model.stream(messages, temperature=0.3)
        return stream, rewritten_query, retrieval_result
