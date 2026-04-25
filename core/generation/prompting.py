"""用于查询改写与答案生成的提示词构造器。"""

from __future__ import annotations

from typing import Any


def build_rewrite_messages(query: str, chat_history: list[tuple[str, str]]) -> list[dict[str, str]]:
    """构造用于检索优化的查询改写消息。"""
    # 只保留最近若干轮对话，避免提示词过长并保持改写目标聚焦。
    history = "\n".join([f"{role}: {text}" for role, text in chat_history[-8:]])
    return [
        {
            "role": "system",
            "content": "你是查询优化器。请将用户问题改写为适合检索的单句，不要回答问题本身。",
        },
        {
            "role": "user",
            "content": f"历史对话:\n{history}\n\n当前问题:\n{query}\n\n输出改写后的查询：",
        },
    ]


def build_answer_messages(query: str, context: str, chat_history: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """构造带上下文注入的答案生成消息。"""
    # 将最近的对话上下文一起送入模型，保证回答能延续多轮语境。
    history = "\n".join([f"{role}: {text}" for role, text in chat_history[-8:]])
    return [
        {
            "role": "system",
            "content": "你是实验室知识库助手 LabKB。必须优先依据给定上下文回答；若证据不足明确说不知道；给出简洁回答并附来源。",
        },
        {
            "role": "user",
            "content": f"历史对话:\n{history}\n\n上下文:\n{context}\n\n问题:\n{query}\n\n请作答。",
        },
    ]


def build_context(items: list[dict[str, Any]], max_chars: int = 9000) -> str:
    """从召回结果中构造上下文窗口，并保留来源标记。

    优先使用父块完整文本（metadata.parent_text），降级使用子块文本（content）。
    """
    chunks: list[str] = []
    total = 0
    for it in items:
        # 优先使用父块完整文本，降级使用子块文本
        metadata = (it.get("metadata") or {})
        text = metadata.get("parent_text") or it.get("content", "")
        piece = f"[source:{it.get('source','')}] {text}"
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "\n\n".join(chunks)
