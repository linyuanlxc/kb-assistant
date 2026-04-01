"""顶栏组件：品牌标题、健康状态指示器。"""

from __future__ import annotations

import streamlit as st


def render_topbar(pipeline) -> None:
    """渲染顶部导航栏。"""
    qdrant_ok = pipeline.orchestrator.vector_store.health()
    graph_ok = pipeline.orchestrator.graph_engine.health()
    neo4j_cls = "online" if graph_ok else "offline"
    ready_cls = "online" if qdrant_ok else "offline"

    st.markdown(
        f"""
        <div class="topbar">
            <div class="topbar-title">
                <span class="logo-icon">KB</span>
                KB Assistant V2
                <span class="topbar-subtitle">LightRAG + Multimodal</span>
            </div>
            <div class="topbar-actions">
                <span class="topbar-status">
                    <span class="status-dot online"></span>Qdrant
                </span>
                <span class="topbar-status">
                    <span class="status-dot {neo4j_cls}"></span>Neo4j
                </span>
                <span class="topbar-status">
                    <span class="status-dot {ready_cls}"></span>检索就绪
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
