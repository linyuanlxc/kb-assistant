import os
import sys
import shutil
from pathlib import Path
from uuid import uuid4

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from embeddings.qwen_embedding import QwenEmbeddings

# ====== 配置区 ======
SOURCE_DIR = ROOT_DIR / "kb_source"                       # 原始知识文件目录
PERSIST_DIRECTORY = ROOT_DIR / "data_base" / "vector_db" / "chroma"
COLLECTION_NAME = "default_kb"
RESET_DB = True                               # True = 每次执行都重建整个库；False = 增量添加

SUPPORTED_SUFFIXES = {".md", ".markdown", ".pdf", ".docx", ".doc"}


def collect_files(source_dir: str):
    """递归收集支持的文件。"""
    base = Path(source_dir)
    if not base.exists():
        raise FileNotFoundError(f"知识源目录不存在: {source_dir}")

    files = []
    for path in base.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)

    return sorted(files)


def load_one_file(file_path: Path):
    """
    按文件类型选择 loader。
    - md/markdown: 当普通文本读取
    - pdf: PyPDFLoader
    - docx: Docx2txtLoader，失败时尝试 UnstructuredWordDocumentLoader
    - doc: 走 UnstructuredWordDocumentLoader（需要 unstructured）
    """
    suffix = file_path.suffix.lower()

    if suffix in {".md", ".markdown"}:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()

    elif suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()

    elif suffix == ".docx":
        try:
            loader = Docx2txtLoader(str(file_path))
            docs = loader.load()
        except Exception:
            try:
                from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            except ImportError as e:
                raise RuntimeError(
                    "当前 .docx 用 Docx2txtLoader 读取失败；如需兜底，请安装: pip install -U unstructured"
                ) from e
            loader = UnstructuredWordDocumentLoader(str(file_path), mode="single")
            docs = loader.load()

    elif suffix == ".doc":
        try:
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
        except ImportError as e:
            raise RuntimeError(
                "处理 .doc 需要安装: pip install -U unstructured"
            ) from e
        loader = UnstructuredWordDocumentLoader(str(file_path), mode="single")
        docs = loader.load()

    else:
        raise ValueError(f"不支持的文件类型: {file_path}")

    # 补充 metadata，后续方便做过滤、删除、定位来源
    for doc in docs:
        doc.metadata["source"] = str(file_path)
        doc.metadata["file_name"] = file_path.name
        doc.metadata["file_type"] = suffix

    return docs


def load_all_documents(source_dir: str):
    """加载目录下全部支持的文件。"""
    files = collect_files(source_dir)
    if not files:
        raise ValueError(f"目录 {source_dir} 下没有找到支持的文件类型: {SUPPORTED_SUFFIXES}")

    all_docs = []
    for file_path in files:
        try:
            docs = load_one_file(file_path)
            all_docs.extend(docs)
            print(f"[OK] 已加载: {file_path} -> {len(docs)} 个文档对象")
        except Exception as e:
            print(f"[SKIP] 加载失败: {file_path}\n       原因: {e}")

    if not all_docs:
        raise ValueError("没有成功加载任何文档。")

    return all_docs


def split_documents(documents):
    """
    切块：
    - RecursiveCharacterTextSplitter 是 LangChain 推荐的通用切分器
    - separators 里加入中文标点，中文检索效果通常更稳
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True,
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",   # 零宽空格
            "\uff0c",   # 中文全角逗号
            "\u3001",   # 顿号
            "\uff0e",   # 全角句号
            "\u3002",   # 中文句号
            "",
        ],
    )
    split_docs = splitter.split_documents(documents)

    # 给 chunk 增加 chunk_id
    for i, doc in enumerate(split_docs):
        doc.metadata["chunk_index"] = i

    return split_docs


def reset_db(persist_directory: str):
    """删除旧库，重建。"""
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"[OK] 已删除旧向量库: {persist_directory}")


def build_vector_db():
    # 1) 读取原始文档
    raw_docs = load_all_documents(SOURCE_DIR)
    print(f"[INFO] 原始文档对象总数: {len(raw_docs)}")

    # 2) 切块
    split_docs = split_documents(raw_docs)
    print(f"[INFO] 切块后文档数: {len(split_docs)}")

    # 3) 是否重建
    if RESET_DB:
        reset_db(PERSIST_DIRECTORY)

    # 4) 初始化 Embedding
    embedding = QwenEmbeddings()

    # 5) 初始化/加载 Chroma
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=str(PERSIST_DIRECTORY),
    )

    # 6) 写入向量库
    ids = [str(uuid4()) for _ in range(len(split_docs))]
    vector_store.add_documents(documents=split_docs, ids=ids)

    print("\n========== 建库完成 ==========")
    print(f"知识源目录: {SOURCE_DIR}")
    print(f"向量库存储目录: {PERSIST_DIRECTORY}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"写入 chunk 数量: {len(split_docs)}")


if __name__ == "__main__":
    build_vector_db()