"""数据加载模块（文本 + 图片）。

设计原则：
1. 文本和图片统一输出 `RawDocument`。
2. 通过 checksum 支持增量更新与去重。
3. 保持加载阶段无副作用，方便测试和复用。
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader

from core.types import RawDocument

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".json", ".srt", ".vtt", ".tsv", ".html"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def sha256_file(path: Path) -> str:
    """计算文件 sha256，用于变更检测。"""
    hash_obj = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def _id_from_path(path: Path) -> str:
    """基于绝对路径生成稳定 doc_id。"""
    return str(uuid5(NAMESPACE_URL, str(path.resolve())))


def _load_text(path: Path) -> str:
    """按文件类型加载文本内容。"""
    suffix = path.suffix.lower()

    if suffix in TEXT_SUFFIXES:
        return TextLoader(str(path), encoding="utf-8").load()[0].page_content

    if suffix == ".pdf":
        pages = PyPDFLoader(str(path)).load()
        return "\n\n".join(page.page_content for page in pages)

    if suffix in {".doc", ".docx"}:
        pages = Docx2txtLoader(str(path)).load()
        return "\n\n".join(page.page_content for page in pages)

    return ""


def load_raw_documents(source_dir: Path) -> tuple[list[RawDocument], list[RawDocument]]:
    """扫描目录并返回（文本文档列表，图片资产列表）。"""
    text_docs: list[RawDocument] = []
    image_docs: list[RawDocument] = []

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        checksum = sha256_file(path)
        now = datetime.now()
        doc_id = _id_from_path(path)

        if suffix in TEXT_SUFFIXES or suffix in {".pdf", ".doc", ".docx"}:
            content = _load_text(path)
            if not content.strip():
                continue

            text_docs.append(
                RawDocument(
                    doc_id=doc_id,
                    source_path=str(path),
                    source_type=suffix,
                    modality="text",
                    content=content,
                    title=path.stem,
                    checksum=checksum,
                    created_at=now,
                    updated_at=now,
                    extra_meta={"file_name": path.name},
                )
            )
            continue

        if suffix in IMAGE_SUFFIXES:
            image_docs.append(
                RawDocument(
                    doc_id=doc_id,
                    source_path=str(path),
                    source_type=suffix,
                    modality="image",
                    content=path.name,
                    title=path.stem,
                    checksum=checksum,
                    created_at=now,
                    updated_at=now,
                    extra_meta={"file_name": path.name},
                )
            )

    return text_docs, image_docs
