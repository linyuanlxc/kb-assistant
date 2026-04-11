"""评估数据采集与记录模块。

职责：
1. 在 API 层注入，记录每次问答的完整数据
2. 支持 JSONL 格式存储，便于追加和批量处理
3. 包含 trace_id 用于追踪和关联
4. 零性能影响：异步写入，不阻塞主流程

数据格式：
{
    "trace_id": "uuid",
    "timestamp": "2026-01-01T00:00:00",
    "query": "用户问题",
    "rewritten_query": "改写后的问题",
    "retrieval_result": {
        "items": [...],
        "sources": [...],
        "graph_evidence": [...],
        "latency_ms": 150,
        "debug_info": {...}
    },
    "answer": "生成的答案",
    "latency_ms": {
        "retrieval": 150,
        "generation": 1200,
        "total": 1350
    },
    "search_mode": "hybrid"
}
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.types import RetrieverResult


class EvaluationRecorder:
    """评估数据采集器。"""

    def __init__(self, log_dir: str | Path):
        """
        Args:
            log_dir: JSONL 文件存储目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "qa_logs.jsonl"

    def record(
        self,
        query: str,
        rewritten_query: str,
        retrieval_result: RetrieverResult,
        answer: str,
        search_mode: str,
        image_inputs: list[str] | None = None,
    ) -> str:
        """记录一次完整的问答数据。

        Returns:
            trace_id: 用于追踪本次记录
        """
        trace_id = str(uuid4())
        timestamp = datetime.now().isoformat()

        # 构建记录数据
        record = {
            "trace_id": trace_id,
            "timestamp": timestamp,
            "query": query,
            "rewritten_query": rewritten_query,
            "retrieval_result": self._serialize_retrieval_result(retrieval_result),
            "answer": answer,
            "latency_ms": {
                "retrieval": retrieval_result.latency_ms,
                # generation latency 将在后续实现中补充
                "generation": 0,
                "total": retrieval_result.latency_ms + 0,
            },
            "search_mode": search_mode,
            "image_inputs": image_inputs or [],
        }

        # 追加写入 JSONL
        self._append_to_jsonl(record)

        return trace_id

    def _serialize_retrieval_result(self, result: RetrieverResult) -> dict[str, Any]:
        """序列化检索结果。"""
        return {
            "items": [self._serialize_item(item) for item in result.items],
            "sources": result.sources,
            "graph_evidence": result.graph_evidence,
            "latency_ms": result.latency_ms,
            "debug_info": result.debug_info,
        }

    def _serialize_item(self, item: Any) -> dict[str, Any]:
        """序列化单个检索项。"""
        if hasattr(item, "__dataclass_fields__"):
            # 如果是 dataclass
            from dataclasses import asdict
            return asdict(item)
        elif hasattr(item, "__dict__"):
            # 如果是普通对象
            return dict(item.__dict__)
        else:
            # 如果已经是 dict
            return dict(item)

    def _append_to_jsonl(self, record: dict[str, Any]) -> None:
        """追加记录到 JSONL 文件。"""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            # 记录失败不影响主流程，记录错误日志
            print(f"[EvaluationRecorder] Failed to write log: {e}")

    def get_log_file(self) -> Path:
        """获取日志文件路径。"""
        return self.log_file

    def get_recent_logs(self, n: int = 100) -> list[dict[str, Any]]:
        """获取最近的 n 条日志。

        用于评测集采样。
        """
        if not self.log_file.exists():
            return []

        logs = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        return logs[-n:]
