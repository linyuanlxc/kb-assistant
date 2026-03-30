"""增量索引构建所用的 manifest 持久化工具。"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from core.types import IndexManifest


def load_manifest(path: Path) -> IndexManifest:
    """从磁盘加载 manifest，不存在时返回空 manifest。"""
    if not path.exists():
        return IndexManifest()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return IndexManifest(**data)


def save_manifest(path: Path, manifest: IndexManifest) -> None:
    """以 UTF-8 JSON 格式持久化 manifest。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.last_sync_at = datetime.now().isoformat()
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, ensure_ascii=False, indent=2)


def register_version(manifest: IndexManifest, note: str) -> None:
    """为当前索引运行追加版本记录。"""
    manifest.versions.append({"time": datetime.now().isoformat(), "note": note})
