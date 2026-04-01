"""聊天面板组件：消息列表渲染、流式输出、图片预览。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st


def render_chat_messages(
    messages: list[dict[str, Any]],
) -> None:
    """渲染历史消息列表。"""
    if not messages:
        _render_welcome()
        return

    for msg in messages:
        role = msg.get("role", "human")
        content = msg.get("content", "")
        image_path = msg.get("image_path")
        sources = msg.get("sources", [])
        latency_ms = msg.get("latency_ms")
        rewritten = msg.get("rewritten")

        if role == "human":
            _render_human_msg(content, image_path)
        elif role == "ai":
            _render_ai_msg(content, sources, latency_ms, rewritten)


def render_streaming_answer(
    stream,
    rewritten: str,
    latency_ms: float,
    sources: list[str],
) -> str:
    """渲染流式 AI 回答并返回完整文本。"""
    with st.chat_message("ai"):
        answer_text = st.write_stream(stream)

        meta_parts = []
        if rewritten:
            meta_parts.append(f'<span class="rewrite-tag">改写: {rewritten}</span>')
        if latency_ms:
            meta_parts.append(f'<span class="latency-tag">检索: {latency_ms:.0f}ms</span>')
        if meta_parts:
            st.markdown(
                f'<div class="msg-meta">{"".join(meta_parts)}</div>',
                unsafe_allow_html=True,
            )

        if sources:
            source_tags = []
            for src in sources[:6]:
                name = Path(src).name if src else src
                is_img = any(src.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp"))
                label = "[img]" if is_img else ""
                source_tags.append(f'<span class="source-tag">{label} {name}</span>')
            st.markdown(
                f'<div class="source-tags">{"".join(source_tags)}</div>',
                unsafe_allow_html=True,
            )

    return answer_text


def _render_welcome() -> None:
    """渲染欢迎页面。"""
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">KB</div>
            <div class="empty-title">KB Assistant V2</div>
            <div class="empty-desc">
                基于 LightRAG + 多模态检索的个人知识库助手<br><br>
                混合检索 -- 向量 + 关键词 + 图谱<br>
                多模态 -- 以文搜图 / 以图搜图<br>
                多轮对话 -- 上下文感知的智能问答<br><br>
                在下方输入问题开始使用，或上传图片进行相似搜索
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_human_msg(content: str, image_path: str | None) -> None:
    """渲染用户消息。"""
    with st.chat_message("human"):
        if image_path:
            img_p = Path(image_path)
            if img_p.exists():
                st.image(str(img_p), width=200)
        st.markdown(content)


def _render_ai_msg(
    content: str,
    sources: list[str] | None = None,
    latency_ms: float | None = None,
    rewritten: str | None = None,
) -> None:
    """渲染 AI 消息。"""
    with st.chat_message("ai"):
        st.markdown(content)

        meta_parts = []
        if rewritten:
            meta_parts.append(f'<span class="rewrite-tag">改写: {rewritten}</span>')
        if latency_ms:
            meta_parts.append(f'<span class="latency-tag">检索: {latency_ms:.0f}ms</span>')
        if meta_parts:
            st.markdown(
                f'<div class="msg-meta">{"".join(meta_parts)}</div>',
                unsafe_allow_html=True,
            )

        if sources:
            source_tags = []
            for src in sources[:6]:
                name = Path(src).name if src else src
                is_img = any(src.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp"))
                label = "[img]" if is_img else ""
                source_tags.append(f'<span class="source-tag">{label} {name}</span>')
            st.markdown(
                f'<div class="source-tags">{"".join(source_tags)}</div>',
                unsafe_allow_html=True,
            )
