"""BM25 稀疏索引封装。

Stores serialized records on disk and rebuilds runtime BM25 model on load.
"""

from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from core.types import ChunkRecord, RetrieverItem


class BM25Indexer:
    """Simple BM25 indexer with local JSON persistence."""

    def __init__(self, index_file: Path):
        self.index_file = index_file
        self.corpus: list[list[str]] = []
        self.records: list[dict] = []
        self.bm25 = None

    def build(self, chunks: list[ChunkRecord]) -> None:
        """Build BM25 model from chunk records."""
        self.records = [
            {
                "chunk_id": c.chunk_id,
                "text": c.chunk_text,
                "source": c.metadata.get("source_path", ""),
                "metadata": c.metadata,
            }
            for c in chunks
        ]
        self.corpus = [r["text"].lower().split() for r in self.records]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def save(self) -> None:
        """Save sparse index payload."""
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        with self.index_file.open("w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False)

    def load(self) -> None:
        """Load sparse index payload and rebuild runtime BM25 object."""
        if not self.index_file.exists():
            return
        with self.index_file.open("r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.corpus = [r["text"].lower().split() for r in self.records]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def search(self, query: str, top_k: int = 10) -> list[RetrieverItem]:
        """Search BM25 top-k items."""
        if not self.bm25:
            return []
        q = query.lower().split()
        scores = self.bm25.get_scores(q)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        items: list[RetrieverItem] = []
        for idx, score in indexed:
            r = self.records[idx]
            items.append(
                RetrieverItem(
                    item_id=r["chunk_id"],
                    content=r["text"],
                    source=r["source"],
                    score=float(score),
                    metadata=r.get("metadata", {}),
                )
            )
        return items
