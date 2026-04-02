"""来源面板组件：文本来源 + 图片缩略图 + Debug 信息。"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import streamlit as st


def render_source_panel(
    items: list[Any] | None = None,
    debug_info: dict[str, Any] | None = None,
    debug: bool = False,
) -> None:
    if not items:
        st.caption("No results.")
        return

    text_items, image_items = [], []
    for item in items:
        meta = item.metadata if hasattr(item, "metadata") else {}
        if meta.get("modality") == "image":
            image_items.append(item)
        else:
            text_items.append(item)

    # ── Text Sources ──
    if text_items:
        st.markdown(
            '<div class="source-section-title">Text Sources</div>',
            unsafe_allow_html=True,
        )
        for item in text_items[:8]:
            _render_text_card(item)

    # ── Image Sources ──
    if image_items:
        st.markdown(
            '<div class="source-section-title">Image Sources</div>',
            unsafe_allow_html=True,
        )
        _render_image_grid(image_items[:8])

    # ── Debug ──
    if debug and debug_info:
        st.markdown("---")
        with st.expander("Debug Info", expanded=False):
            st.json(debug_info)


def _render_text_card(item: Any) -> None:
    meta = item.metadata if hasattr(item, "metadata") else {}
    source = item.source if hasattr(item, "source") else meta.get("source", "")
    content = item.content if hasattr(item, "content") else meta.get("content", "")
    score = item.score if hasattr(item, "score") else meta.get("score", 0)

    file_name = Path(source).name if source else "Unknown"
    summary = content[:120].replace("\n", " ").replace("<", "&lt;").replace(">", "&gt;")
    summary = summary + ("..." if len(content) > 120 else "")
    safe_source = html.escape(source) if source else ""

    st.markdown(
        f"""
        <div class="text-source-card">
            <div class="card-header">
                <span class="card-icon">T</span>
                <span class="card-path" title="{safe_source}">{file_name}</span>
                <span class="card-score">{score:.3f}</span>
            </div>
            <div class="card-summary">{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_image_grid(items: list[Any]) -> None:
    cols = st.columns(2)
    for idx, item in enumerate(items):
        meta = item.metadata if hasattr(item, "metadata") else {}
        source = item.source if hasattr(item, "source") else meta.get("source", "")
        score = item.score if hasattr(item, "score") else meta.get("score", 0)
        file_name = html.escape(Path(source).name) if source else "image"

        with cols[idx % 2]:
            source_path = Path(source) if source else None
            if source_path and source_path.exists():
                st.image(str(source_path), use_container_width=True)
                st.caption(f"**{file_name}** | {score:.3f}")
            else:
                st.markdown(
                    f"""
                    <div class="image-result-card">
                        <div style="aspect-ratio:4/3;display:flex;align-items:center;
                                    justify-content:center;background:#1C2128;
                                    color:#8B949E;font-size:12px;">N/A</div>
                        <div class="card-overlay">
                            <span class="card-name">{file_name}</span>
                            <span class="score-badge">{score:.3f}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
