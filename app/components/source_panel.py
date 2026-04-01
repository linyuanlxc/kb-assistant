"""来源面板组件：文本来源卡片 + 图片缩略图网格 + 点击放大查看。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st


def render_source_panel(
    sources: list[str] | None,
    items: list[Any] | None = None,
    debug_info: dict[str, Any] | None = None,
    debug: bool = False,
) -> None:
    """渲染右侧来源面板。"""
    if not items and not sources:
        _render_empty()
        return

    text_items = []
    image_items = []

    if items:
        for item in items:
            meta = item.metadata if hasattr(item, "metadata") else {}
            if meta.get("modality") == "image":
                image_items.append(item)
            else:
                text_items.append(item)

    total = len(text_items) + len(image_items)
    st.markdown(
        f"""
        <div class="source-panel-header">
            <h3>检索来源 <span class="count-badge">{total}</span></h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if text_items:
        st.markdown(
            '<div class="source-section-title">文本来源</div>',
            unsafe_allow_html=True,
        )
        for item in text_items[:8]:
            _render_text_card(item)

    if image_items:
        st.markdown(
            '<div class="source-section-title">图片来源</div>',
            unsafe_allow_html=True,
        )
        _render_image_grid(image_items[:8])

    if debug and debug_info:
        st.markdown("---")
        st.markdown(
            '<div class="source-section-title">调试信息</div>',
            unsafe_allow_html=True,
        )
        with st.expander("查看 Debug 详情", expanded=False):
            st.json(debug_info)


def _render_empty() -> None:
    """渲染空状态。"""
    st.markdown(
        """
        <div class="source-panel-header">
            <h3>检索来源</h3>
        </div>
        <div class="empty-state">
            <div class="empty-icon">--</div>
            <div class="empty-title">暂无检索结果</div>
            <div class="empty-desc">发送问题后，检索到的来源<br>和图片将在此处展示</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_text_card(item: Any) -> None:
    """渲染单条文本来源卡片。"""
    meta = item.metadata if hasattr(item, "metadata") else {}
    source = item.source if hasattr(item, "source") else meta.get("source", "")
    content = item.content if hasattr(item, "content") else meta.get("content", "")
    score = item.score if hasattr(item, "score") else meta.get("score", 0)

    file_name = Path(source).name if source else "未知来源"
    summary = content[:120].replace("\n", " ") + ("..." if len(content) > 120 else "")

    st.markdown(
        f"""
        <div class="text-source-card" onclick="void(0)">
            <div class="card-header">
                <span class="card-icon">T</span>
                <span class="card-path" title="{source}">{file_name}</span>
                <span class="card-score">{score:.3f}</span>
            </div>
            <div class="card-summary">{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_image_grid(items: list[Any]) -> None:
    """渲染图片缩略图网格，支持点击放大。"""
    cols = st.columns(2)
    for idx, item in enumerate(items):
        col = cols[idx % 2]
        meta = item.metadata if hasattr(item, "metadata") else {}
        source = item.source if hasattr(item, "source") else meta.get("source", "")
        score = item.score if hasattr(item, "score") else meta.get("score", 0)
        file_name = Path(source).name if source else "image"

        with col:
            source_path = Path(source) if source else None
            if source_path and source_path.exists():
                with st.popover("", use_container_width=True):
                    st.image(str(source_path), use_container_width=True)
                    st.caption(f"来源: {source}")
                    st.caption(f"相似度: {score:.3f}")

                st.image(
                    str(source_path),
                    use_container_width=True,
                )
                st.caption(f"**{file_name}** | {score:.3f}")
            else:
                st.markdown(
                    f"""
                    <div class="image-result-card">
                        <div style="aspect-ratio:4/3;display:flex;align-items:center;justify-content:center;
                                    background:var(--bg-card);color:var(--text-muted);font-size:12px;">
                            图片不可用
                        </div>
                        <div class="card-overlay">
                            <span class="card-name">{file_name}</span>
                            <span class="score-badge">{score:.3f}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
