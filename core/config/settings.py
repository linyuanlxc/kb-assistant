"""运行时配置加载。

设计目标：
1. 统一管理配置来源，避免分散读取。
2. 提供类型化配置对象，降低运行时配置错误。
3. 支持环境变量覆盖，便于部署。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
PATH_KEYS = {
    "source_dir",
    "index_dir",
    "runtime_dir",
    "model_registry_path",
    "bm25_index_file",
    "manifest_path",
    "model_cache_dir",
    "log_dir",
    "log_file",
}


@dataclass
class AppSettings:
    """应用配置对象（索引与服务阶段共用）。"""

    # HuggingFace 镜像（国内环境加速下载）
    hf_endpoint: str = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    hf_offline: bool = os.getenv("HF_HUB_OFFLINE", "0").lower() in ("1", "true")

    source_dir: Path = ROOT_DIR / "kb_source"
    index_dir: Path = ROOT_DIR / "data_base"
    runtime_dir: Path = ROOT_DIR / "runtime"
    model_cache_dir: Path = ROOT_DIR / "runtime" / "model_cache"
    log_dir: Path = ROOT_DIR / "runtime" / "logs"
    log_file: Path = ROOT_DIR / "runtime" / "logs" / "app.log"

    # 向量库（Qdrant）
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")

    # 图数据库（Neo4j）s
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "neo4j")

    debug_rag: bool = os.getenv("DEBUG_RAG", "false").lower() == "true"
    rerank_enabled: bool = os.getenv("RERANK_ENABLED", "true").lower() == "true"
    rerank_top_k_multiplier: int = 3
    model_registry_path: Path = ROOT_DIR / "core" / "config" / "model_registry.yaml"

    # 集合与文件路径
    text_collection: str = "kb_text_chunks"
    image_collection: str = "kb_image_assets"
    parent_collection: str = "kb_parent_docs"
    bm25_index_file: Path = ROOT_DIR / "data_base" / "bm25_index.json"
    manifest_path: Path = ROOT_DIR / "data_base" / "index_manifest.json"

    # 融合检索默认权重
    retrieval_weights: dict[str, float] = field(
        default_factory=lambda: {
            "text_dense": 0.35,
            "bm25": 0.20,
            "graph": 0.25,
            "image_clip": 0.20,
        }
    )


def _to_path(value: Any) -> Path:
    """把配置值转换为 Path，支持相对路径自动挂到项目根目录。"""
    path_value = Path(str(value))
    if path_value.is_absolute():
        return path_value
    return ROOT_DIR / path_value


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 配置；文件不存在时返回空字典。"""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data


def load_settings() -> AppSettings:
    """加载配置并应用 `default.yaml` 覆盖。"""
    settings = AppSettings()
    cfg_path = ROOT_DIR / "core" / "config" / "default.yaml"
    cfg = load_yaml(cfg_path)

    for key, value in cfg.items():
        if not hasattr(settings, key):
            continue
        if key in PATH_KEYS:
            setattr(settings, key, _to_path(value))
        else:
            setattr(settings, key, value)

    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)

    # 设置 HuggingFace 镜像（影响所有 SentenceTransformer / Transformers 模型下载）
    if settings.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
    # 离线模式：跳过联网验证，直接使用本地缓存
    if settings.hf_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("SENTENCE_TRANSFORMERS_OFFLINE", "1")

    return settings
