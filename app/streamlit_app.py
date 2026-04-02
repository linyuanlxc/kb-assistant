"""KB Assistant V2 - Streamlit 应用入口。

布局参考豆包/元宝等主流 AI 对话产品：
- 左侧边栏：检索模式、参数、系统信息（可折叠）
- 主区域：聊天消息 + 底部统一输入框
- 来源面板：expander 展示，位于输入框上方
- 暗色主题，由 config.toml + theme.css 共同控制
"""

from __future__ import annotations

import base64
import html as _html
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

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

_CSS_PATH = Path(__file__).resolve().parent / "styles" / "theme.css"


def _inject_css() -> None:
    """注入自定义 CSS + JS（布局重排、自动刷新状态、侧边栏控制）。"""
    if _CSS_PATH.exists():
        st.html(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>")

    # JS: 侧边栏折叠控制 + 状态自动刷新
    st.html("""<script>
    (function() {
        /* ── 侧边栏折叠按钮 ── */
        function ensureSidebarToggle() {
            var sidebar = document.querySelector('section[data-testid="stSidebar"]');
            if (!sidebar || sidebar.querySelector('.sidebar-toggle-btn')) return;

            var overlay = document.createElement('button');
            overlay.type = 'button';
            overlay.className = 'sidebar-toggle-btn';
            overlay.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"/></svg>';
            overlay.addEventListener('click', function() {
                var arrow = sidebar.querySelector('.sidebar-toggle-btn svg');
                if (sidebar.style.display === 'none') {
                    sidebar.style.display = '';
                    arrow.style.transform = '';
                } else {
                    sidebar.style.display = 'none';
                    arrow.style.transform = 'rotate(180deg)';
                }
            });
            sidebar.insertBefore(overlay, sidebar.firstChild);
        }

        /* ── Composer 布局重排 ── */
        function syncComposerUi() {
            var container = document.querySelector('.stChatInputContainer');
            if (!container) return;
            var inner = container.querySelector(':scope > div');
            if (!inner) return;
            var chatInput = inner.querySelector('[data-testid="stChatInput"]') || inner.querySelector('.stChatInput');
            if (!chatInput) return;

            var preview = document.querySelector('.upload-preview-inline');
            if (preview && preview.parentElement !== inner) {
                inner.insertBefore(preview, chatInput);
            }
            inner.classList.toggle('has-upload-preview', !!preview);

            var toolbar = inner.querySelector('.composer-toolbar');
            if (!toolbar) {
                toolbar = document.createElement('div');
                toolbar.className = 'composer-toolbar';
                inner.appendChild(toolbar);
            }
            var uploader = document.querySelector('div[data-testid="stFileUploader"]');
            if (uploader && uploader.parentElement !== toolbar) {
                toolbar.appendChild(uploader);
            }

            var removeMarker = document.querySelector('.composer-hidden-actions-anchor');
            var hiddenRemoveButton = null;
            if (removeMarker) {
                var actionRoot = removeMarker.closest('div[data-testid="stVerticalBlock"]');
                hiddenRemoveButton = actionRoot ? actionRoot.querySelector('button') : null;
            }
            if (preview) {
                var removeTrigger = preview.querySelector('.preview-remove-trigger');
                if (removeTrigger && hiddenRemoveButton && !removeTrigger.dataset.bound) {
                    removeTrigger.addEventListener('click', function() { hiddenRemoveButton.click(); });
                    removeTrigger.dataset.bound = '1';
                }
            }
        }

        ensureSidebarToggle();
        syncComposerUi();
        new MutationObserver(function() {
            ensureSidebarToggle();
            syncComposerUi();
        }).observe(document.body, { childList: true, subtree: true });
    })();
    </script>""")


def _bootstrap_pipeline() -> RAGPipeline:
    """初始化 RAG 流水线。"""
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
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password,
    )
    orchestrator = RetrievalOrchestrator(
        settings=settings, registry=registry,
        vector_store=vector_store, bm25=bm25, graph_engine=graph_engine,
    )
    return RAGPipeline(settings=settings, orchestrator=orchestrator, registry=registry)


def _uploaded_file_storage_path(uploaded: BinaryIO) -> Path:
    """为上传图片生成唯一文件名，避免同名覆盖。"""
    temp_dir = ROOT_DIR / "runtime" / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    original_name = getattr(uploaded, "name", "image")
    suffix = Path(original_name).suffix.lower() or ".png"
    stem = re.sub(r"[^0-9A-Za-z_-]+", "-", Path(original_name).stem).strip("-") or "image"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return temp_dir / f"{stem}-{timestamp}-{uuid4().hex[:8]}{suffix}"


def _render_image_upload() -> BinaryIO | None:
    """渲染隐藏上传器 + 预览卡片（JS 会把上传器移到输入框内部工具栏）。"""
    uploader_key = f"image_uploader_{st.session_state.image_uploader_nonce}"
    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key=uploader_key,
    )

    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        safe_name = _html.escape(uploaded.name)
        mime_type = getattr(uploaded, "type", None) or "image/png"
        preview_data = base64.b64encode(image_bytes).decode("ascii")
        st.markdown(
            f"""<div class="upload-preview-inline">
                <div class="preview-thumb-wrap">
                    <img class="preview-thumb" src="data:{mime_type};base64,{preview_data}" alt="{safe_name}">
                </div>
                <div class="preview-copy">
                    <div class="preview-name-small">{safe_name}</div>
                    <div class="preview-meta-small">已选择图片，发送时会和当前文本一起提交</div>
                </div>
                <button type="button" class="preview-remove-trigger" aria-label="删除已选择图片">×</button>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="composer-hidden-actions-anchor"></div>', unsafe_allow_html=True)
        if st.button("Remove selected image", key=f"remove_image_{uploader_key}"):
            st.session_state.image_uploader_nonce += 1
            st.rerun()

    return uploaded


def _save_uploaded(uploaded: BinaryIO | None) -> list[str]:
    """将上传图片保存并返回路径列表。"""
    if uploaded is None:
        return []
    img_file = _uploaded_file_storage_path(uploaded)
    img_file.write_bytes(uploaded.getvalue())
    return [str(img_file)]


def main() -> None:
    logger = setup_logging()

    st.set_page_config(
        page_title="KB Assistant V2",
        page_icon="KB",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_css()

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = _bootstrap_pipeline()
        logger.info("pipeline bootstrapped")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "latest_retrieval" not in st.session_state:
        st.session_state.latest_retrieval = None
    if "image_uploader_nonce" not in st.session_state:
        st.session_state.image_uploader_nonce = 0

    # ── Sidebar ──
    with st.sidebar:
        mode, debug, top_k = render_sidebar(st.session_state.pipeline)
        st.markdown("---")
        if st.button("New Chat", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.latest_retrieval = None
            st.session_state.image_uploader_nonce += 1
            st.rerun()

    # ── Main: Chat + Upload + Source ──
    chat_placeholder = st.empty()
    with chat_placeholder.container():
        render_chat_messages(st.session_state.messages)

    uploaded = _render_image_upload()

    prompt = st.chat_input("输入问题，支持配图一起发送")
    if prompt:
        saved_paths = _save_uploaded(uploaded)
        if not st.session_state.messages:
            chat_placeholder.empty()

        user_msg: dict = {"role": "human", "content": prompt}
        if saved_paths:
            user_msg["image_path"] = saved_paths[0]
        st.session_state.messages.append(user_msg)

        with st.chat_message("human"):
            if saved_paths:
                p = Path(saved_paths[0])
                if p.exists():
                    st.image(str(p), width=220)
            st.markdown(prompt)

        req = RetrieverRequest(
            query=prompt,
            chat_history=[(m.get("role", ""), m.get("content", ""))
                          for m in st.session_state.messages],
            modality=mode,
            image_inputs=saved_paths,
            top_k=top_k,
            debug=debug,
        )

        stream, rewritten, ret = st.session_state.pipeline.answer_stream(req)
        answer_text = render_streaming_answer(
            stream=stream, rewritten=rewritten,
            latency_ms=ret.latency_ms, sources=ret.sources,
        )

        ai_msg = {
            "role": "ai", "content": answer_text,
            "sources": ret.sources, "latency_ms": ret.latency_ms,
            "rewritten": rewritten,
        }
        st.session_state.messages.append(ai_msg)
        st.session_state.latest_retrieval = {
            "items": ret.items, "debug_info": ret.debug_info, "debug": debug,
        }

        st.session_state.image_uploader_nonce += 1
        st.rerun()

    ret_data = st.session_state.latest_retrieval
    if ret_data and ret_data["items"]:
        with st.expander(f"Retrieval Sources ({len(ret_data['items'])})", expanded=False):
            render_source_panel(
                items=ret_data["items"],
                debug_info=ret_data["debug_info"],
                debug=ret_data["debug"],
            )


if __name__ == "__main__":
    main()
