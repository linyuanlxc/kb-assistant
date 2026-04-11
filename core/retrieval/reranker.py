"""重排模块。

在多路召回融合之后、送入 LLM 生成之前，使用 Cross-Encoder
对候选文档进行精细化排序，提升最终上下文质量。

支持的实现：
- ``llm``: 利用已有的 Chat LLM 做零样本文档相关性打分（无需额外模型）。
- ``cross-encoder``: 本地 Cross-Encoder 模型（如 BGE-Reranker），精度更高。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from core.types import RetrieverItem


class BaseReranker(ABC):
    """重排器抽象接口。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        items: list[RetrieverItem],
        top_k: int | None = None,
    ) -> list[RetrieverItem]:
        """对候选列表重新排序并返回 top_k 条。"""
        ...

    @abstractmethod
    def health(self) -> bool:
        """检查重排器是否可用。"""
        ...


class LLMReranker(BaseReranker):
    """基于 LLM 的重排器。

    利用已有的 Chat LLM 对每个候选文档打分（0-10），
    无需额外下载模型，适合快速验证场景。
    """

    def __init__(self, model_cfg: dict[str, Any]):
        from openai import OpenAI

        api_key = os.getenv(model_cfg.get("api_key_env", ""), "")
        self.client = OpenAI(
            api_key=api_key,
            base_url=model_cfg.get("base_url", ""),
            timeout=20,
        )
        self.model = model_cfg.get("model", "glm-4-flash")

    def _score_single(self, query: str, content: str) -> float:
        """让 LLM 为单条文档与查询的相关性打分。"""
        prompt = (
            "你是一个文档相关性评判专家。请判断以下文档内容与用户问题的相关性。\n"
            "只输出一个 0 到 10 的整数分数，不要输出其他内容。\n"
            "0 = 完全无关，10 = 高度相关且直接回答问题。\n\n"
            f"用户问题：{query}\n\n"
            f"文档内容：{content[:800]}\n\n"
            "相关性分数："
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4,
            )
            text = (resp.choices[0].message.content or "").strip()
            score = float("".join(c for c in text if c.isdigit() or c == "."))
            return min(max(score, 0.0), 10.0)
        except Exception:
            return 0.0

    def rerank(
        self,
        query: str,
        items: list[RetrieverItem],
        top_k: int | None = None,
    ) -> list[RetrieverItem]:
        if not items:
            return items

        # 对有文本内容的候选项打分
        for item in items:
            content = item.content.strip()
            if content:
                item.score = self._score_single(query, content)

        items.sort(key=lambda x: x.score, reverse=True)
        return items[:top_k] if top_k else items

    def health(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False


class CrossEncoderReranker(BaseReranker):
    """本地 Cross-Encoder 重排器。

    使用 FlagEmbedding / sentence-transformers 的 Cross-Encoder 模型
    进行高精度相关性打分。
    """

    def __init__(self, model_name: str, batch_size: int = 32, max_length: int = 512):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._model

    def rerank(
        self,
        query: str,
        items: list[RetrieverItem],
        top_k: int | None = None,
    ) -> list[RetrieverItem]:
        if not items:
            return items

        pairs = [(query, item.content.strip()) for item in items if item.content.strip()]
        valid_items = [item for item in items if item.content.strip()]

        if not pairs:
            return items

        model = self._get_model()
        scores = model.predict(pairs, batch_size=self.batch_size)

        for item, score in zip(valid_items, scores):
            item.score = float(score)

        valid_items.sort(key=lambda x: x.score, reverse=True)
        result = valid_items[:top_k] if top_k else valid_items

        # 追加被跳过的无内容项
        skipped = [item for item in items if not item.content.strip()]
        return result + skipped

    def health(self) -> bool:
        try:
            self._get_model()
            return True
        except Exception:
            return False


def build_reranker(cfg: dict[str, Any]) -> BaseReranker | None:
    """根据配置构造重排器实例。返回 None 表示不启用重排。"""
    if not cfg or not cfg.get("enabled", False):
        return None

    provider = cfg.get("provider", "llm")
    if provider == "cross-encoder":
        model_name = cfg.get("model", "BAAI/bge-reranker-v2-m3")
        return CrossEncoderReranker(
            model_name=model_name,
            batch_size=cfg.get("batch_size", 32),
            max_length=cfg.get("max_length", 512),
        )
    # 默认使用 LLM 重排
    model_key = cfg.get("model_key", "fast_model")
    from core.providers.model_provider import ModelRegistry

    registry_path = cfg.get("registry_path")
    if registry_path:
        from pathlib import Path

        registry = ModelRegistry(Path(registry_path))
    else:
        from core.config.settings import AppSettings

        registry = ModelRegistry(AppSettings.model_registry_path)
    model_cfg = registry.providers.get(model_key, {})
    if not model_cfg:
        return None
    return LLMReranker(model_cfg)
