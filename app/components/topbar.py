"""顶栏组件：品牌标题 + 健康状态。"""

from __future__ import annotations

import streamlit as st


def render_topbar(pipeline) -> None:
    qdrant_ok = pipeline.orchestrator.vector_store.health()
    graph_ok = pipeline.orchestrator.graph_engine.health()

    st.markdown(
        f"""
        <div class="topbar">
            <div class="topbar-brand">
                <span class="logo-icon">KB</span>
                <div class="topbar-copy">
                    <div class="topbar-title">KB Assistant</div>
                    <div class="topbar-subtitle">Knowledge chat workspace</div>
                </div>
            </div>
            <div class="topbar-actions">
                <span class="topbar-status">
                    <span class="status-dot online"></span>Qdrant
                </span>
                <span class="topbar-status">
                    <span class="status-dot {"online" if graph_ok else "offline"}"></span>Neo4j
                </span>
                <span class="topbar-status">
                    <span class="status-dot {"online" if qdrant_ok else "offline"}"></span>Ready
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
