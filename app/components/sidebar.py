"""侧边栏组件：主题切换、检索模式、Top K、Debug、系统信息。"""

from __future__ import annotations

import streamlit as st

from core.types import SearchMode


def render_sidebar() -> tuple[SearchMode, bool, int]:
    """渲染侧边栏控件，返回 (检索模式, debug开关, top_k)。"""

    # ── 主题切换 ──
    st.markdown('<div class="sidebar-label">外观</div>', unsafe_allow_html=True)
    theme_options = {"light": "浅色模式", "dark": "深色模式"}
    current = st.session_state.get("theme", "dark")
    new_theme_key = st.radio(
        "主题",
        list(theme_options.keys()),
        format_func=lambda x: theme_options[x],
        label_visibility="collapsed",
        horizontal=True,
        index=0 if current == "light" else 1,
    )
    if new_theme_key != current:
        st.session_state.theme = new_theme_key
        st.rerun()
    st.markdown("---")

    # ── 检索模式 ──
    st.markdown('<div class="sidebar-label">检索模式</div>', unsafe_allow_html=True)

    mode_map = {
        "混合检索": SearchMode.HYBRID,
        "纯文本": SearchMode.TEXT_ONLY,
        "多模态": SearchMode.MULTIMODAL,
        "图谱优先": SearchMode.GRAPH_FIRST,
    }
    mode_desc = {
        "混合检索": "Dense + BM25 + Graph + Image",
        "纯文本": "仅向量 + 关键词检索",
        "多模态": "支持图片输入的跨模态检索",
        "图谱优先": "图谱关系优先召回",
    }

    selected_label = st.radio(
        "检索模式",
        list(mode_map.keys()),
        label_visibility="collapsed",
    )
    mode = mode_map[selected_label]

    st.caption(mode_desc[selected_label])
    st.markdown("---")

    # ── Top K ──
    st.markdown('<div class="sidebar-label">检索参数</div>', unsafe_allow_html=True)
    top_k = st.slider("Top K", min_value=3, max_value=20, value=10, step=1)
    st.markdown("---")

    # ── Debug ──
    debug = st.toggle("Debug 模式", value=False, help="展示检索诊断信息")
    st.markdown("---")

    # ── 系统信息 ──
    st.markdown('<div class="sidebar-label">系统信息</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sys-info-card">
            <div>模型: <b>GLM-4-Flash</b></div>
            <div>Embedding: <b>CLIP ViT-B/32</b></div>
            <div>向量维度: <b>512d</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return mode, debug, top_k
