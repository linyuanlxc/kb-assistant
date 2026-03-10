from typing import List
from langchain_core.embeddings import Embeddings
import os
from openai import OpenAI
import streamlit as st

class QwenEmbeddings(Embeddings):
    def __init__(self):
        api_key=st.secrets.get("QWEN_API_KEY")
        if not api_key:
            raise ValueError("请提供千问API密钥")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """

        result = []
        for i in range(0, len(texts), 64):
            embeddings = self.client.embeddings.create(
                model="text-embedding-v3",
                input=texts[i:i+64],
                dimensions=1024
            )
            result.extend([embeddings.embedding for embeddings in embeddings.data])
        return result

    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            texts (str): 要生成 embedding 的文本.

        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """

        return self.embed_documents([text])[0]