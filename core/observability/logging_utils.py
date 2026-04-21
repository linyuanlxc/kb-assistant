"""结构化日志工具。

输出 JSON 日志，支持双通道：
1. 控制台（实时查看）
2. 文件（按天滚动归档）
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Iterator

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LOG_FILE = ROOT_DIR / "runtime" / "logs" / "app.log"


class JsonFormatter(logging.Formatter):
    """将标准日志记录格式化为 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "trace_id"):
            payload["trace_id"] = record.trace_id
        if hasattr(record, "extra_data"):
            payload["extra_data"] = record.extra_data
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """初始化统一 logger（重复调用时复用已存在 handler）。"""
    logger = logging.getLogger("lab_kb")
    logger.setLevel(level)
    if logger.handlers:
        return logger

    formatter = JsonFormatter()

    # 1) 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2) 文件输出（每天滚动，最多保留 14 天）
    log_file = Path(os.getenv("KB_LOG_FILE", str(DEFAULT_LOG_FILE)))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


@contextmanager
def trace_span(logger: logging.Logger, stage: str) -> Iterator[str]:
    """记录阶段耗时并自动注入 trace_id。"""
    trace_id = uuid.uuid4().hex[:16]
    start = time.perf_counter()
    logger.info(f"{stage} started", extra={"trace_id": trace_id})
    try:
        yield trace_id
    finally:
        cost_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"{stage} finished",
            extra={"trace_id": trace_id, "extra_data": {"latency_ms": round(cost_ms, 2)}},
        )
