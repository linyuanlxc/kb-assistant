"""聊天面板组件：消息列表渲染、流式输出。"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import streamlit as st


def render_chat_messages(messages: list[dict[str, Any]]) -> None:
    if not messages:
        _render_welcome()
        return
    for msg in messages:
        role = msg.get("role", "human")
        if role == "human":
            with st.chat_message("human"):
                image_path = msg.get("image_path")
                if image_path:
                    p = Path(image_path)
                    if p.exists():
                        st.image(str(p), width=200)
                st.markdown(msg.get("content", ""))
        elif role == "ai":
            with st.chat_message("ai"):
                st.markdown(msg.get("content", ""))
                _render_meta(msg.get("rewritten"), msg.get("latency_ms"))
                _render_source_tags(msg.get("sources", []))


def render_streaming_answer(stream, rewritten: str, latency_ms: float, sources: list[str]) -> str:
    with st.chat_message("ai"):
        answer_text = st.write_stream(stream)
        _render_meta(rewritten, latency_ms)
        _render_source_tags(sources)
    return answer_text


def _render_meta(rewritten: str | None, latency_ms: float | None) -> None:
    parts = []
    if rewritten:
        safe = html.escape(rewritten)
        parts.append(f'<span class="rewrite-tag">Rewrite: {safe}</span>')
    if latency_ms:
        parts.append(f'<span class="latency-tag">{latency_ms:.0f}ms</span>')
    if parts:
        st.markdown(
            f'<div class="msg-meta">{"".join(parts)}</div>',
            unsafe_allow_html=True,
        )


def _render_source_tags(sources: list[str]) -> None:
    if not sources:
        return
    tags = []
    for src in sources[:6]:
        name = html.escape(Path(src).name) if src else src
        is_img = any(src.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp"))
        label = "[IMG]" if is_img else "[DOC]"
        tags.append(f'<span class="source-tag">{label} {name}</span>')
    st.markdown(
        f'<div class="source-tags">{"".join(tags)}</div>',
        unsafe_allow_html=True,
    )


def _render_welcome() -> None:
    st.markdown(
        """
        <section class="welcome-shell">
            <div class="welcome-badge">KB Assistant</div>
            <h1 class="welcome-title">围绕知识库的 AI 对话工作区</h1>
            <p class="welcome-copy">
                选择左侧检索模式后直接提问，系统会在同一个界面里完成召回、生成和来源展示。
            </p>
            <div class="welcome-grid">
                <div class="welcome-card">
                    <div class="welcome-card-title">Hybrid Search</div>
                    <div class="welcome-card-copy">适合大多数问题，综合向量、关键词和图谱检索。</div>
                </div>
                <div class="welcome-card">
                    <div class="welcome-card-title">Multimodal</div>
                    <div class="welcome-card-copy">支持上传图片，再结合文本知识一起分析与回答。</div>
                </div>
                <div class="welcome-card">
                    <div class="welcome-card-title">Traceable Answers</div>
                    <div class="welcome-card-copy">回答后可继续查看来源片段、图像结果和调试信息。</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
