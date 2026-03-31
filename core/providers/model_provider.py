"""模型提供层抽象。

本模块统一封装以下能力：
- 对话生成（同步 / 流式）
- 文本向量化
- 图像向量化（CLIP）

并把模型缓存目录固定到项目 `runtime/model_cache`，便于环境复现与清理。
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CACHE = ROOT_DIR / "runtime" / "model_cache"

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


def _prepare_local_model_cache() -> Path:
    """设置模型下载缓存目录到项目下并返回目录路径。"""
    cache_dir = Path(os.getenv("KB_MODEL_CACHE", str(DEFAULT_MODEL_CACHE)))
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 统一 HuggingFace / Transformers / SentenceTransformers 缓存目录。
    os.environ.setdefault("HF_HOME", str(cache_dir / "hf_home"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_dir / "sentence_transformers"))

    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["SENTENCE_TRANSFORMERS_HOME"]).mkdir(parents=True, exist_ok=True)
    return cache_dir


class ChatProvider:
    """OpenAI 兼容对话模型封装。"""

    def __init__(self, model_cfg: dict[str, Any]):
        api_key = os.getenv(model_cfg["api_key_env"], "")
        self.client = OpenAI(api_key=api_key, base_url=model_cfg["base_url"], timeout=30)
        self.model = model_cfg["model"]

    def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """执行非流式对话生成。"""
        resp = self.client.chat.completions.create(model=self.model, messages=messages, stream=False, **kwargs)
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[dict[str, Any]], **kwargs: Any):
        """执行流式对话生成，并逐段产出内容。"""
        resp = self.client.chat.completions.create(model=self.model, messages=messages, stream=True, **kwargs)
        for chunk in resp:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta


class EmbeddingProvider:
    """统一的文本 / 图像向量化接口。"""

    def __init__(self, text_cfg: dict[str, Any], image_cfg: dict[str, Any]):
        self.cache_dir = _prepare_local_model_cache()

        self.text_provider = text_cfg.get("provider", "api")
        self.text_dimensions = text_cfg.get("dimensions", 1024)

        if self.text_provider == "clip":
            # CLIP 本地模型：文本和图片共享同一向量空间
            self.text_model_name = text_cfg.get("model", "")
            self._text_model = None
            if SentenceTransformer and self.text_model_name:
                self._text_model = SentenceTransformer(self.text_model_name)
        else:
            # API 方式（OpenAI-compatible）
            api_key = os.getenv(text_cfg.get("api_key_env", ""), "")
            self.text_client = OpenAI(api_key=api_key, base_url=text_cfg.get("base_url", ""), timeout=20)
            self.text_model = text_cfg.get("model", "")

        self.image_model_name = image_cfg.get("model", "")
        self._image_model = None
        if SentenceTransformer and self.image_model_name:
            self._image_model = SentenceTransformer(self.image_model_name)

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """使用配置好的文本 embedding 模型批量向量化文本。"""
        if not texts:
            return []

        if self.text_provider == "clip":
            if not self._text_model:
                return [[0.0] * self.text_dimensions for _ in texts]
            return self._text_model.encode(texts, normalize_embeddings=True).tolist()

        result: list[list[float]] = []
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.text_client.embeddings.create(
                model=self.text_model,
                input=batch,
                dimensions=self.text_dimensions,
            )
            result.extend([item.embedding for item in resp.data])

        return result

    def embed_query(self, text: str) -> list[float]:
        """向量化单条查询文本。"""
        return self.embed_text([text])[0]

    def embed_image_paths(self, image_paths: list[str]) -> list[list[float]]:
        """使用 CLIP 模型向量化图片文件。

        当 CLIP 不可用时返回零向量，保证系统可降级运行。
        """
        if not image_paths:
            return []
        if not self._image_model:
            return [[0.0] * 512 for _ in image_paths]
        return self._image_model.encode(image_paths, normalize_embeddings=True).tolist()


class ModelRegistry:
    """加载模型注册表，并构造 provider 实例。"""

    def __init__(self, registry_path: Path):
        with registry_path.open("r", encoding="utf-8") as file:
            self._cfg = yaml.safe_load(file) or {}

    @property
    def providers(self) -> dict[str, Any]:
        return self._cfg.get("providers", {})

    @property
    def embeddings(self) -> dict[str, Any]:
        return self._cfg.get("embeddings", {})

    def build_chat(self, key: str) -> ChatProvider:
        return ChatProvider(self.providers[key])

    def build_embedding(self) -> EmbeddingProvider:
        return EmbeddingProvider(self.embeddings["text"], self.embeddings["image"])

    @staticmethod
    def image_bytes_to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
        """将原始图片字节转换为 base64 data URL。"""
        return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
