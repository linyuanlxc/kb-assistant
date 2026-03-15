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

# =========================
# 日志配置
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("kb_assistant")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from embeddings.qwen_embedding import QwenEmbeddings
from langchain_community.vectorstores import Chroma


def get_retriever():
    start = time.perf_counter()
    try:
        logger.info("开始初始化 retriever")

        # 定义 Embeddings
        embedding = QwenEmbeddings()

        # 向量数据库持久化路径
        persist_directory = ROOT_DIR / "data_base" / "vector_db" / "chroma"
        logger.info("Chroma 路径: %s", persist_directory)

        # 加载数据库
        vectordb = Chroma(
            collection_name="default_kb",
            persist_directory=str(persist_directory),
            embedding_function=embedding
        )

        retriever = vectordb.as_retriever(search_kwargs={"k": 2})
        logger.info("retriever 初始化完成，耗时 %.3fs", time.perf_counter() - start)
        return retriever

    except Exception:
        logger.exception("初始化 retriever 失败")
        raise


def combine_docs(docs):
    try:
        context_docs = docs["context"]
        logger.info("开始拼接上下文，共 %d 个文档片段", len(context_docs))
        combined = "\n\n".join(doc.page_content for doc in context_docs)
        logger.info("上下文拼接完成，总长度 %d 字符", len(combined))
        return combined
    except Exception:
        logger.exception("拼接上下文失败")
        raise


def get_qa_history_chain():
    start = time.perf_counter()
    try:
        logger.info("开始初始化 QA 链")

        retriever = get_retriever()

        llm = ChatOpenAI(
            api_key=st.secrets.get("QWEN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name="qwen3.5-flash",
        )
        logger.info("LLM 初始化完成，模型: qwen3.5-flash")

        condense_question_system_template = (
            "请根据聊天记录总结用户最近的问题，"
            "如果没有多余的聊天记录则返回用户的问题。"
        )
        condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

        retrieve_docs = RunnableBranch(
            (
                lambda x: not x.get("chat_history", False),
                (lambda x: x["input"]) | retriever,
            ),
            condense_question_prompt | llm | StrOutputParser() | retriever,
        )

        system_prompt = (
            "你是一个问答任务的助手。 "
            "请使用检索到的上下文片段回答这个问题。 "
            "如果你不知道答案就说不知道。 "
            "请使用简洁的话语回答用户。"
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )

        qa_chain = (
            RunnablePassthrough().assign(context=combine_docs)
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        qa_history_chain = (
            RunnablePassthrough()
            .assign(context=retrieve_docs)
            .assign(answer=qa_chain)
        )

        logger.info("QA 链初始化完成，耗时 %.3fs", time.perf_counter() - start)
        return qa_history_chain

    except Exception:
        logger.exception("初始化 QA 链失败")
        raise


def gen_response(chain, input, chat_history):
    start = time.perf_counter()
    try:
        logger.info(
            "开始生成回答 | 输入长度=%d | 历史消息数=%d | 用户问题=%s",
            len(input),
            len(chat_history),
            input[:100]
        )

        response = chain.stream({
            "input": input,
            "chat_history": chat_history
        })

        yielded = False
        for res in response:
            if "answer" in res.keys():
                yielded = True
                yield res["answer"]

        logger.info(
            "回答生成完成 | 是否有输出=%s | 总耗时=%.3fs",
            yielded,
            time.perf_counter() - start
        )

    except Exception:
        logger.exception("生成回答失败")
        raise


# Streamlit 应用程序界面
def main():
    app_start = time.perf_counter()
    logger.info("应用启动")

    try:
        st.markdown('### 🦜🔗 动手学大模型应用开发')

        # 用于跟踪对话历史
        if "messages" not in st.session_state:
            st.session_state.messages = []
            logger.info("初始化 session_state.messages")

        # 存储检索问答链
        if "qa_history_chain" not in st.session_state:
            logger.info("session_state 中未发现 QA 链，开始创建")
            st.session_state.qa_history_chain = get_qa_history_chain()

        messages = st.container(height=550)

        # 显示整个对话历史
        logger.info("渲染历史消息，共 %d 条", len(st.session_state.messages))
        for message in st.session_state.messages:
            with messages.chat_message(message[0]):
                st.write(message[1])

        if prompt := st.chat_input("Say something"):
            logger.info("收到用户输入: %s", prompt[:100])

            # 将用户输入添加到对话历史中
            st.session_state.messages.append(("human", prompt))
            with messages.chat_message("human"):
                st.write(prompt)

            answer = gen_response(
                chain=st.session_state.qa_history_chain,
                input=prompt,
                chat_history=st.session_state.messages
            )

            with messages.chat_message("ai"):
                output = st.write_stream(answer)

            st.session_state.messages.append(("ai", output))
            logger.info("本轮对话完成，当前总消息数=%d", len(st.session_state.messages))

        logger.info("本次页面执行完成，总耗时 %.3fs", time.perf_counter() - app_start)

    except Exception:
        logger.exception("main() 执行失败")
        st.error("应用运行出错，请查看日志。")
        raise


if __name__ == "__main__":
    main()