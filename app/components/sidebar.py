"""侧边栏组件：检索模式、参数、系统信息。"""

from __future__ import annotations

import streamlit as st

from core.types import SearchMode


def render_sidebar(pipeline) -> tuple[SearchMode, bool, int]:
    # ── System Status (auto-refresh on each rerun) ──
    qdrant_ok = pipeline.orchestrator.vector_store.health()
    graph_ok = pipeline.orchestrator.graph_engine.health()
    ready_ok = qdrant_ok and graph_ok

    st.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand-mark">KB</div>
            <div class="sidebar-brand-copy">
                <div class="sidebar-brand-title">KB Assistant</div>
                <div class="sidebar-brand-subtitle">Knowledge Chat Workspace</div>
            </div>
        </div>
        <div class="sidebar-status-stack">
            <div class="sidebar-status-item">
                <span class="status-dot {"online" if qdrant_ok else "offline"}"></span>
                <span>Qdrant</span>
            </div>
            <div class="sidebar-status-item">
                <span class="status-dot {"online" if graph_ok else "offline"}"></span>
                <span>Neo4j</span>
            </div>
            <div class="sidebar-status-item">
                <span class="status-dot {"online" if ready_ok else "offline"}"></span>
                <span>Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Search Mode ──
    st.markdown('<div class="sidebar-label">Search Mode</div>', unsafe_allow_html=True)
    mode_map = {
        "Hybrid": SearchMode.HYBRID,
        "Text Only": SearchMode.TEXT_ONLY,
        "Multimodal": SearchMode.MULTIMODAL,
        "Graph First": SearchMode.GRAPH_FIRST,
    }
    mode_desc = {
        "Hybrid": "Dense + BM25 + Graph + Image",
        "Text Only": "Vector + keyword only",
        "Multimodal": "Cross-modal image search",
        "Graph First": "Knowledge graph priority",
    }
    selected = st.radio("Mode", list(mode_map.keys()), label_visibility="collapsed")
    st.caption(mode_desc[selected])
    st.markdown("---")

    # ── Params ──
    st.markdown('<div class="sidebar-label">Parameters</div>', unsafe_allow_html=True)
    top_k = st.slider("Top K", min_value=3, max_value=20, value=10, step=1)
    st.markdown("---")

    # ── Debug ──
    debug = st.toggle("Debug", value=False)
    st.markdown("---")

    # ── System Info ──
    st.markdown('<div class="sidebar-label">System</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sys-info-card">
            <div>LLM: <b>GLM-4-Flash</b></div>
            <div>Embedding: <b>CLIP ViT-B/32</b></div>
            <div>Dimension: <b>512d</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return mode_map[selected], debug, top_k
