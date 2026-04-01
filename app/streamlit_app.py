"""KB Assistant V2 - Streamlit 应用入口。

功能说明：
- 深色/浅色主题切换
- 三栏布局（聊天 + 来源面板）
- 混合检索（Qdrant + BM25 + LightRAG）
- 支持图片上传的多模态检索（以文搜图、以图搜图）
- 流式回答生成 + 检索来源可视化
- 带检索诊断信息的调试面板
"""

from __future__ import annotations

import base64
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

from app.components.chat_panel import render_chat_messages, render_streaming_answer
from app.components.sidebar import render_sidebar
from app.components.source_panel import render_source_panel
from app.components.topbar import render_topbar


# ── CSS 注入 ──
_CSS_PATH = Path(__file__).resolve().parent / "styles" / "theme.css"


def _inject_css(theme: str) -> None:
    """注入自定义主题 CSS。

    策略：根据 theme 值直接生成对应变量集的 CSS，
    不依赖 JS/data-theme 属性切换（避免 iframe 跨域问题）。
    """
    if not _CSS_PATH.exists():
        return
    css_text = _CSS_PATH.read_text(encoding="utf-8")

    if theme == "light":
        # 替换 :root 中的暗色变量为亮色变量
        light_vars = {
            "--bg-primary": "#F8F9FC",
            "--bg-secondary": "#FFFFFF",
            "--bg-card": "#FFFFFF",
            "--bg-hover": "#F1F3F8",
            "--border-color": "#E2E5EF",
            "--text-primary": "#1F2937",
            "--text-secondary": "#4B5563",
            "--text-muted": "#9CA3AF",
            "--accent-blue": "#2563EB",
            "--accent-green": "#059669",
            "--accent-red": "#DC2626",
            "--accent-yellow": "#D97706",
            "--accent-teal": "#0D9488",
            "--accent-blue-dim": "rgba(37, 99, 235, 0.1)",
            "--topbar-bg": "var(--bg-secondary)",
            "--msg-ai-bg": "#F3F4F6",
            "--msg-ai-border": "var(--border-color)",
            "--input-bg": "#F3F4F6",
            "--card-hover-border": "var(--accent-blue)",
            "--shadow-card": "0 1px 4px rgba(0, 0, 0, 0.08)",
            "--shadow-glow": "0 0 16px rgba(37, 99, 235, 0.1)",
        }
        var_overrides = "\n".join(f"  {k}: {v};" for k, v in light_vars.items())
        theme_css = f":root {{\n{var_overrides}\n}}"
        # 注入亮色变量覆盖 + 通用样式
        st.markdown(
            f"<style>{theme_css}\n{css_text}</style>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)


# ── Pipeline 初始化 ──


def _bootstrap_pipeline() -> RAGPipeline:
    """初始化运行时依赖，返回可用的 RAG 流水线。"""
    settings = load_settings()
    registry = ModelRegistry(settings.model_registry_path)

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


# ── 图片上传与预览 ──


def _render_image_upload() -> tuple[object, list[str]]:
    """渲染图片上传区域和预览。"""
    uploaded = st.file_uploader(
        "上传图片（可选，用于多模态检索）",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key="image_uploader",
    )

    image_paths: list[str] = []
    if uploaded is not None:
        temp_dir = ROOT_DIR / "runtime" / "uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        img_file = temp_dir / uploaded.name
        img_file.write_bytes(uploaded.getvalue())
        image_paths.append(str(img_file))

        b64 = base64.b64encode(uploaded.getvalue()).decode("utf-8")
        st.markdown(
            f"""
            <div class="upload-preview-area">
                <div class="upload-preview-thumb">
                    <img src="data:image/jpeg;base64,{b64}" alt="preview">
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"已选择: {uploaded.name}（用于图片相似搜索）")

    return uploaded, image_paths


def _save_uploaded_image(uploaded: object) -> list[str]:
    """将上传图片保存并返回路径。"""
    if uploaded is None:
        return []

    temp_dir = ROOT_DIR / "runtime" / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    img_file = temp_dir / uploaded.name
    img_file.write_bytes(uploaded.getvalue())
    return [str(img_file)]


# ── 主入口 ──


def main() -> None:
    """主 UI 入口。"""
    logger = setup_logging()

    st.set_page_config(
        page_title="KB Assistant V2",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 主题状态
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

    # 注入主题 CSS
    _inject_css(st.session_state.theme)

    # 初始化 pipeline
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = _bootstrap_pipeline()
        logger.info("pipeline bootstrapped")

    # 消息历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 最新检索结果
    if "latest_retrieval" not in st.session_state:
        st.session_state.latest_retrieval = None

    # 顶栏
    render_topbar(st.session_state.pipeline)

    # 侧边栏
    with st.sidebar:
        mode, debug, top_k = render_sidebar()

        st.markdown("---")
        if st.button("新对话", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.latest_retrieval = None
            st.rerun()

    # 主体三栏布局
    chat_col, source_col = st.columns([5, 2])

    with chat_col:
        render_chat_messages(st.session_state.messages)

        # 图片上传
        with st.expander("上传图片进行搜索", expanded=False):
            uploaded, image_paths = _render_image_upload()

        # 聊天输入
        if prompt := st.chat_input("输入问题，或上传图片搜索相似内容..."):
            saved_paths = _save_uploaded_image(uploaded) if uploaded else []

            user_msg = {"role": "human", "content": prompt}
            if saved_paths:
                user_msg["image_path"] = saved_paths[0]
            st.session_state.messages.append(user_msg)

            with st.chat_message("human"):
                if saved_paths:
                    p = Path(saved_paths[0])
                    if p.exists():
                        st.image(str(p), width=200)
                st.markdown(prompt)

            req = RetrieverRequest(
                query=prompt,
                chat_history=[(m.get("role", ""), m.get("content", "")) for m in st.session_state.messages],
                modality=mode,
                image_inputs=saved_paths,
                top_k=top_k,
                debug=debug,
            )

            stream, rewritten, ret = st.session_state.pipeline.answer_stream(req)

            answer_text = render_streaming_answer(
                stream=stream,
                rewritten=rewritten,
                latency_ms=ret.latency_ms,
                sources=ret.sources,
            )

            ai_msg = {
                "role": "ai",
                "content": answer_text,
                "sources": ret.sources,
                "latency_ms": ret.latency_ms,
                "rewritten": rewritten,
            }
            st.session_state.messages.append(ai_msg)

            st.session_state.latest_retrieval = {
                "items": ret.items,
                "debug_info": ret.debug_info,
                "debug": debug,
            }

    with source_col:
        ret_data = st.session_state.latest_retrieval
        render_source_panel(
            sources=ret_data["items"][0].source if ret_data and ret_data["items"] else None,
            items=ret_data["items"] if ret_data else None,
            debug_info=ret_data["debug_info"] if ret_data else None,
            debug=ret_data["debug"] if ret_data else False,
        )


if __name__ == "__main__":
    main()
