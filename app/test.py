import streamlit as st
from langchain_openai import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
import sys
from pathlib import Path
import logging
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from embeddings.qwen_embedding import QwenEmbeddings
from langchain_community.vectorstores import Chroma

def get_retriever():
    start = time.perf_counter()
    try:

        # 定义 Embeddings
        embedding = QwenEmbeddings()

        # 向量数据库持久化路径
        persist_directory = ROOT_DIR / "data_base" / "vector_db" / "chroma"

        # 加载数据库
        vectordb = Chroma(
            collection_name="default_kb",
            persist_directory=str(persist_directory),
            embedding_function=embedding
        )

        print(vectordb._collection.count())

        retriever = vectordb.as_retriever(search_kwargs={"k": 2})
        return retriever

    except Exception:
        raise

retriever = get_retriever()
question = "解释下c中的指针吧"
docs = retriever.get_relevant_documents(question)
print(docs)
# for i, doc in enumerate(docs):
#     print(f"文档 {i+1} 内容:\n{doc.page_content}\n")


