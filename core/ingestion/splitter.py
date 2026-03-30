"""父子分块模块。

- 父块用于语义完整性。
- 子块用于检索粒度。
"""

from __future__ import annotations

from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.types import ChunkRecord, RawDocument


def _token_estimate(text: str) -> int:
    """基于字符数估算 token 数，作为轻量预算指标。"""
    return max(1, len(text) // 4)


def split_parent_child(
    docs: list[RawDocument],
    parent_chunk_size: int = 1500,
    parent_overlap: int = 120,
    child_chunk_size: int = 380,
    child_overlap: int = 70,
) -> tuple[list[ChunkRecord], dict[str, str]]:
    """将文档切分为子块，并返回父块文本映射。"""
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    child_chunks: list[ChunkRecord] = []
    parent_store: dict[str, str] = {}

    for doc in docs:
        parents = parent_splitter.split_text(doc.content)
        for parent_index, parent_text in enumerate(parents):
            parent_id = f"{doc.doc_id}:p:{parent_index}"
            parent_store[parent_id] = parent_text

            # 子块承接父块 ID，实现检索后回溯父文档。
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
