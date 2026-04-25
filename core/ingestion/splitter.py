"""父子分块模块。

- 父块用于语义完整性。
- 子块用于检索粒度。
- 支持结构化 Markdown 文档（按标题边界切分父块）。
"""

from __future__ import annotations

from typing import Any

from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.types import ChunkRecord, RawDocument


def _token_estimate(text: str) -> int:
    """基于字符数估算 token 数，作为轻量预算指标。"""
    return max(1, len(text) // 4)


def _collect_structured_parents(
    nodes: list[dict[str, Any]],
    parent_chunk_size: int,
) -> list[tuple[int, str]]:
    """将结构化节点树按标题边界分组，生成父块文本列表。

    策略：深度优先展平，累积节点文本，直到超过 parent_chunk_size 为止；
    此时将已累积的节点合并为一个父块，然后从下一个节点开始新的父块。
    """
    results: list[tuple[int, str]] = []
    flat_sections: list[str] = []

    def _flatten(node: dict[str, Any]) -> None:
        parts: list[str] = []
        if node.get("title"):
            parts.append("#" * node["level"] + " " + node["title"])
        if node.get("content"):
            parts.append(node["content"])
        flat_sections.append("\n\n".join(parts))
        for child in node.get("children", []):
            _flatten(child)

    for node in nodes:
        _flatten(node)

    # 按 parent_chunk_size 分组
    current_text = ""
    for section in flat_sections:
        if not current_text:
            current_text = section
        elif len(current_text) + len(section) + 2 <= parent_chunk_size:
            current_text += "\n\n" + section
        else:
            results.append((len(results), current_text))
            current_text = section
    if current_text:
        results.append((len(results), current_text))

    return results


def split_parent_child(
    docs: list[RawDocument],
    parent_chunk_size: int = 1500,
    parent_overlap: int = 120,
    child_chunk_size: int = 380,
    child_overlap: int = 70,
    structured_docs: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[list[ChunkRecord], dict[str, str]]:
    """将文档切分为子块，并返回父块文本映射。

    Args:
        docs: 原始文档列表。
        parent_chunk_size: 父块最大字符数。
        parent_overlap: 父块重叠字符数。
        child_chunk_size: 子块最大字符数。
        child_overlap: 子块重叠字符数。
        structured_docs: 结构化文档映射 {doc_id: 层级节点列表}，
                         由 md_parser.parse_markdown() 生成。
    """
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    child_chunks: list[ChunkRecord] = []
    parent_store: dict[str, str] = {}

    for doc in docs:
        # 优先使用结构化切分
        if structured_docs and doc.doc_id in structured_docs:
            parent_entries = _collect_structured_parents(
                structured_docs[doc.doc_id],
                parent_chunk_size,
            )
            for parent_index, parent_text in parent_entries:
                parent_id = f"{doc.doc_id}:p:{parent_index}"
                parent_store[parent_id] = parent_text

                children = child_splitter.split_text(parent_text)
                for child_index, child_text in enumerate(children):
                    chunk_id = f"{doc.doc_id}:c:{parent_index}:{child_index}:{uuid4().hex[:8]}"
                    child_chunks.append(
                        ChunkRecord(
                            doc_id=doc.doc_id,
                            parent_id=parent_id,
                            chunk_id=chunk_id,
                            chunk_text=child_text,
                            chunk_index=child_index,
                            token_count=_token_estimate(child_text),
                            metadata={
                                "source_path": doc.source_path,
                                "file_type": doc.source_type,
                                "file_name": doc.extra_meta.get("file_name", ""),
                                "checksum": doc.checksum,
                                "title": doc.title,
                                "modality": doc.modality,
                            },
                        )
                    )
            continue

        # 非结构化文档：使用原有纯文本切分逻辑
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        parents = parent_splitter.split_text(doc.content)
        for parent_index, parent_text in enumerate(parents):
            parent_id = f"{doc.doc_id}:p:{parent_index}"
            parent_store[parent_id] = parent_text

            children = child_splitter.split_text(parent_text)
            for child_index, child_text in enumerate(children):
                chunk_id = f"{doc.doc_id}:c:{parent_index}:{child_index}:{uuid4().hex[:8]}"
                child_chunks.append(
                    ChunkRecord(
                        doc_id=doc.doc_id,
                        parent_id=parent_id,
                        chunk_id=chunk_id,
                        chunk_text=child_text,
                        chunk_index=child_index,
                        token_count=_token_estimate(child_text),
                        metadata={
                            "source_path": doc.source_path,
                            "file_type": doc.source_type,
                            "file_name": doc.extra_meta.get("file_name", ""),
                            "checksum": doc.checksum,
                            "title": doc.title,
                            "modality": doc.modality,
                        },
                    )
                )

    return child_chunks, parent_store
