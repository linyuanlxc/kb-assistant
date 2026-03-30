"""核心数据契约定义。

该模块统一描述索引、检索、生成三个阶段共享的数据结构，
避免跨模块的隐式字段依赖，提升可维护性和可测试性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SearchMode(str, Enum):
    """检索模式枚举。"""

    TEXT_ONLY = "text_only"
    MULTIMODAL = "multimodal"
    GRAPH_FIRST = "graph_first"
    HYBRID = "hybrid"


@dataclass
class RawDocument:
    """统一原始文档对象（文本/图片共用）。"""

    doc_id: str
    source_path: str
    source_type: str
    modality: str
    content: str
    title: str | None = None
    checksum: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
    extra_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkRecord:
    """文本子块记录（含父子关联）。"""

    doc_id: str
    parent_id: str
    chunk_id: str
    chunk_text: str
    chunk_index: int
    token_count: int
    metadata: dict[str, Any]


@dataclass
class RetrieverItem:
    """单条检索候选结果。"""

    item_id: str
    content: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrieverRequest:
    """检索请求契约。"""

    query: str
    chat_history: list[tuple[str, str]]
    filters: dict[str, Any] = field(default_factory=dict)
    modality: SearchMode = SearchMode.HYBRID
    image_inputs: list[str] = field(default_factory=list)
    top_k: int = 10
    debug: bool = False


@dataclass
class RetrieverResult:
    """检索结果契约。"""

    items: list[RetrieverItem]
    scores: dict[str, float]
    sources: list[str]
    graph_evidence: list[dict[str, Any]]
    latency_ms: float
    debug_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexManifest:
    """索引清单：用于增量更新和版本追踪。"""

    docs: dict[str, dict[str, Any]] = field(default_factory=dict)
    assets: dict[str, dict[str, Any]] = field(default_factory=dict)
    versions: list[dict[str, Any]] = field(default_factory=list)
    last_sync_at: str = ""
