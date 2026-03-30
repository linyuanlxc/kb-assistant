"""KB Assistant V2 的 Streamlit 应用入口。

功能说明：
- 混合检索（Qdrant + BM25 + LightRAG）
- 支持图片上传的多模态检索
- 流式回答生成
- 带检索诊断信息的调试面板
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from core.config.settings import load_settings
from core.indexing.bm25_index import BM25Indexer
from core.indexing.vector_store import QdrantVectorStoreAdapter
from core.observability.logging_utils import setup_logging
from core.orchestration.pipeline import RAGPipeline, RetrievalOrchestrator
from core.providers.model_provider import ModelRegistry
from core.retrieval.graph_retriever import LightRAGGraphEngine
from core.types import RetrieverRequest, SearchMode


def _bootstrap_pipeline() -> RAGPipeline:
    """初始化运行时依赖，并返回可直接使用的 RAG 流水线。"""
    settings = load_settings()
    registry = ModelRegistry(settings.model_registry_path)

    # 先构建底层检索组件，再统一交给编排层组装，避免 UI 层直接接触细节实现。
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
    return RAGPipeline(settings=settings, orchestrator=orchestrator, registry=registry)


def _render_sidebar() -> tuple[SearchMode, bool, int]:
    """渲染侧边栏控件，并返回当前选择的运行参数。"""
    st.sidebar.header("运行参数")
    mode_label = st.sidebar.selectbox(
        "检索模式",
        ["hybrid", "text_only", "multimodal", "graph_first"],
        index=0,
    )
    top_k = st.sidebar.slider("Top K", min_value=3, max_value=20, value=10, step=1)
    debug = st.sidebar.toggle("DEBUG_RAG", value=False)
    return SearchMode(mode_label), debug, top_k


def main() -> None:
    """主 UI 入口。

    Streamlit 会话状态保存：
    - `messages`：聊天历史
    - `pipeline`：已初始化的流水线单例
    """
    logger = setup_logging()
    st.set_page_config(page_title="KB Assistant V2", layout="wide")
    st.title("KB Assistant V2 · LightRAG + Multimodal")

    if "pipeline" not in st.session_state:
        # 仅在首次打开页面时初始化一次，后续复用同一实例。
        st.session_state.pipeline = _bootstrap_pipeline()
        logger.info("pipeline bootstrapped")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    mode, debug, top_k = _render_sidebar()

    uploaded = st.file_uploader(
        "上传图片（可选，用于多模态检索）",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )

    chat_box = st.container(height=520)
    for role, content in st.session_state.messages:
        with chat_box.chat_message(role):
            st.write(content)

    if prompt := st.chat_input("请输入问题"):
        st.session_state.messages.append(("human", prompt))
        with chat_box.chat_message("human"):
            st.write(prompt)

        image_inputs = []
        if uploaded is not None:
            # 将上传图片落盘，供多模态检索链路复用同一路径输入。
            temp_path = ROOT_DIR / "runtime" / "uploads"
            temp_path.mkdir(parents=True, exist_ok=True)
            img_file = temp_path / uploaded.name
            img_file.write_bytes(uploaded.getvalue())
            image_inputs.append(str(img_file))

        req = RetrieverRequest(
            query=prompt,
            chat_history=st.session_state.messages,
            modality=mode,
            image_inputs=image_inputs,
            top_k=top_k,
            debug=debug,
        )

        stream, rewritten, ret = st.session_state.pipeline.answer_stream(req)

        with chat_box.chat_message("ai"):
            # 保持流式输出不变，只补充调试信息展示。
            answer_text = st.write_stream(stream)
            st.caption(f"改写结果：{rewritten}")
            st.caption(f"检索耗时：{ret.latency_ms} ms")
            if ret.sources:
                st.caption("来源：" + " | ".join(ret.sources[:6]))
            if debug:
                st.json(ret.debug_info)

        st.session_state.messages.append(("ai", answer_text))


if __name__ == "__main__":
    main()

