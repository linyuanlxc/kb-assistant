from typing import List
from langchain_core.embeddings import Embeddings
from openai import OpenAI
import streamlit as st
import logging
import time

logger = logging.getLogger("kb_assistant")


class QwenEmbeddings(Embeddings):
    def __init__(self):
        api_key = st.secrets.get("QWEN_API_KEY")
        if not api_key:
            raise ValueError("请提供千问API密钥")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=20.0,
            max_retries=1,
        )
        self.model = "text-embedding-v3"
        self.dimensions = 1024

        logger.info(
            "QwenEmbeddings 初始化完成 | model=%s | dimensions=%s",
            self.model,
            self.dimensions,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            logger.info("embed_documents 收到空输入")
            return []

        total_start = time.perf_counter()
        logger.info("embed_documents 开始 | 文本数=%d", len(texts))

        result: List[List[float]] = []
        try:
            for i in range(0, len(texts), 10):
                batch = texts[i:i + 10]
                batch_start = time.perf_counter()

                logger.info(
                    "embedding batch 开始 | batch_index=%d | batch_size=%d | first_text_len=%d",
                    i // 10,
                    len(batch),
                    len(batch[0]) if batch else 0,
                )

                resp = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions,
                )

                batch_embeddings = [item.embedding for item in resp.data]
                result.extend(batch_embeddings)

                logger.info(
                    "embedding batch 完成 | batch_index=%d | 耗时=%.3fs",
                    i // 10,
                    time.perf_counter() - batch_start,
                )

            logger.info(
                "embed_documents 完成 | 文本数=%d | 总耗时=%.3fs",
                len(texts),
                time.perf_counter() - total_start,
            )
            return result

        except Exception:
            logger.exception("embed_documents 失败")
            raise

    def embed_query(self, text: str) -> List[float]:
        start = time.perf_counter()
        logger.info("embed_query 开始 | 文本长度=%d | 文本=%s", len(text), text[:100])

        try:
            resp = self.client.embeddings.create(
                model=self.model,
                input=[text],
                dimensions=self.dimensions,
            )
            embedding = resp.data[0].embedding

            logger.info(
                "embed_query 完成 | 耗时=%.3fs | 向量维度=%d",
                time.perf_counter() - start,
                len(embedding),
            )
            return embedding

        except Exception:
            logger.exception("embed_query 失败 | 文本=%s", text[:100])
            raise