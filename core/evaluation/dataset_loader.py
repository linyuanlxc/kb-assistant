"""公开测试集加载器。

支持从 Hugging Face 加载以下测试集（均已在 HF Hub 验证可访问）：
- 中文：SuperCLUE-C3-ZH（中文阅读理解）、CRUD-RAG-3QA（事件问答）
- 英文：HotpotQA（多跳问答）、SQuAD v2（抽取式问答）、Amnesty QA（RAGAS 官方示例）

数据模型（每条样本）：
    - question:       测试问题
    - ground_truth:   标准答案
    - golden_docs:    黄金参考文档（该问题对应的相关文档，用于评估检索质量）
    - metadata:       元信息

加载后可提取全量语料库（corpus）：所有样本的 golden_docs 去重合并，
作为向量化的检索源。评估时从 corpus 中检索，与 golden_docs 对比计算检索指标。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset, load_dataset


class DatasetLoader:
    """公开测试集加载器。

    所有数据集路径均已在 Hugging Face Hub 上验证可访问。
    """

    SUPPORTED_DATASETS = {
        # 中文测试集
        "superclue_c3": "TigerResearch/tigerbot-superclue-c3-zh-5k",
        "crud_rag": "AndrewTsai0406/CRUD_RAG_3QA",
        # 英文测试集
        "hotpotqa": "hotpot_qa",
        "squad_v2": "rajpurkar/squad_v2",
        "amnesty_qa": "explodinggradients/amnesty_qa",
    }

    def __init__(
        self,
        dataset_name: Literal["superclue_c3", "crud_rag", "hotpotqa", "squad_v2", "amnesty_qa"],
        subset_size: int | None = None,
        language: Literal["zh", "en"] = "zh",
        cache_dir: str | None = None,
        local_path: str | None = None,
    ):
        """初始化加载器。

        Args:
            dataset_name: 测试集名称
            subset_size: 加载的样本数量（None 表示加载全部）
            language: 语言（影响默认数据集选择）
            cache_dir: Hugging Face 缓存目录
            local_path: 本地 JSONL 文件路径。指定后直接从本地加载，不访问 Hugging Face。
                        默认自动查找 data/eval/{dataset_name}.jsonl
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Supported: {list(self.SUPPORTED_DATASETS.keys())}"
            )

        self.dataset_name = dataset_name
        self.subset_size = subset_size
        self.language = language
        self.cache_dir = cache_dir

        # 本地文件路径：优先使用显式指定的路径，否则查找默认位置
        if local_path:
            self.local_path = Path(local_path)
        else:
            # 自动查找 data/eval/{dataset_name}.jsonl
            default_local = Path(__file__).resolve().parents[3] / "data" / "eval" / f"{dataset_name}.jsonl"
            self.local_path = default_local if default_local.exists() else None

        # 数据集配置映射（所有路径均已在 Hugging Face Hub 验证）
        self._config_mapping = {
            # ── 中文数据集 ──
            "superclue_c3": {
                "name": "TigerResearch/tigerbot-superclue-c3-zh-5k",
                "split": "train",  # 该数据集只有 train split，包含 4792 条
                "field_mapping": {
                    "question": "instruction",
                    "ground_truth": "output",
                    "documents": "input",
                },
                "description": "中文阅读理解，instruction=问题, input=短文, output=答案",
            },
            "crud_rag": {
                "name": "AndrewTsai0406/CRUD_RAG_3QA",
                "split": "train",  # 该数据集只有 train split，包含 3199 条
                "field_mapping": {
                    "question": "questions",
                    "ground_truth": "answers",
                    "documents": "news",  # news1 + news2 + news3 合并为文档
                },
                "description": "基于事件的中文问答，questions=问题, answers=答案, news1/2/3=参考新闻",
            },
            # ── 英文数据集 ──
            "hotpotqa": {
                "name": "hotpot_qa",
                "config": "fullwiki",
                "split": "validation",
                "field_mapping": {
                    "question": "question",
                    "ground_truth": "answer",
                    "documents": "context",  # [[title, text], ...]
                },
                "description": "多跳问答，context 格式为 [[title, text], ...]",
            },
            "squad_v2": {
                "name": "rajpurkar/squad_v2",
                "split": "validation",
                "field_mapping": {
                    "question": "question",
                    "ground_truth": "answers.text",  # 嵌套字段
                    "documents": "context",
                },
                "description": "Stanford 抽取式问答 v2，context=参考段落, answers.text=答案列表",
            },
            "amnesty_qa": {
                "name": "explodinggradients/amnesty_qa",
                "config": "english_v3",
                "split": "eval",  # 只有 20 条，用于快速验证 RAGAS 流程
                "field_mapping": {
                    "question": "user_input",
                    "ground_truth": "reference",
                    "documents": "retrieved_contexts",
                },
                "description": "RAGAS 官方示例数据集，仅 20 条，用于快速验证评估流程",
            },
        }

    def load(self) -> list[dict[str, Any]]:
        """加载并格式化测试集。

        优先从本地 JSONL 文件加载；若本地文件不存在则从 Hugging Face 下载。

        Returns:
            QA 对列表，每个元素包含：
            - question: 用户问题
            - ground_truth: 标准答案（字符串或字符串列表）
            - golden_docs: 黄金参考文档列表（该问题的相关文档，用于评估检索质量）
            - metadata: 额外元信息
        """
        print(f"[DatasetLoader] Loading dataset: {self.dataset_name}")

        # ── 优先从本地 JSONL 加载 ──
        if self.local_path and self.local_path.exists():
            print(f"[DatasetLoader] Loading from local file: {self.local_path}")
            return self._load_from_local()

        # ── 从 Hugging Face 下载 ──
        if self.local_path:
            print(f"[DatasetLoader] Local file not found: {self.local_path}")
            print(f"[DatasetLoader] Falling back to Hugging Face download...")

        try:
            raw_dataset = self._load_raw_dataset()
            print(f"[DatasetLoader] Raw dataset loaded: {len(raw_dataset)} samples")

            formatted_data = self._format_dataset(raw_dataset)
            print(f"[DatasetLoader] Formatted: {len(formatted_data)} valid QA pairs")

            if self.subset_size and self.subset_size < len(formatted_data):
                formatted_data = formatted_data[: self.subset_size]
                print(f"[DatasetLoader] Subset size: {len(formatted_data)}")

            return formatted_data

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {self.dataset_name}: {e}") from e

    def _load_from_local(self) -> list[dict[str, Any]]:
        """从本地 JSONL 文件加载。"""
        data = []
        with open(self.local_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    # 验证必要字段（兼容 documents 和 golden_docs）
                    has_fields = item.get("question") and item.get("ground_truth")
                    has_docs = item.get("golden_docs") or item.get("documents")
                    if has_fields and has_docs:
                        # 统一字段名为 golden_docs
                        if "golden_docs" not in item and "documents" in item:
                            item["golden_docs"] = item.pop("documents")
                        data.append(item)
                except json.JSONDecodeError:
                    continue

        print(f"[DatasetLoader] Loaded {len(data)} samples from local file")

        if self.subset_size and self.subset_size < len(data):
            data = data[: self.subset_size]
            print(f"[DatasetLoader] Subset size: {len(data)}")

        return data

    def _load_raw_dataset(self) -> Dataset:
        """从 Hugging Face 加载原始数据集。"""
        config = self._config_mapping[self.dataset_name]

        load_kwargs = {
            "path": config["name"],
            "split": config["split"],
            "cache_dir": self.cache_dir,
        }

        if "config" in config:
            load_kwargs["name"] = config["config"]

        dataset = load_dataset(**load_kwargs)

        # 过滤空数据
        if self.dataset_name == "crud_rag":
            dataset = dataset.filter(lambda x: bool(x.get("questions")) and bool(x.get("answers")))

        return dataset

    def _format_dataset(self, dataset: Dataset) -> list[dict[str, Any]]:
        """将原始数据集格式化为统一格式。"""
        config = self._config_mapping[self.dataset_name]
        field_mapping = config["field_mapping"]

        formatted_data = []

        for idx, item in enumerate(dataset):
            try:
                question = self._extract_field(item, field_mapping["question"])
                ground_truth = self._extract_field(item, field_mapping["ground_truth"])
                documents = self._extract_documents(item, field_mapping["documents"])

                # 数据集特定后处理
                if self.dataset_name == "crud_rag":
                    question, ground_truth, documents = self._post_process_crud_rag(item)
                    if not question:
                        continue

                # SQuAD v2：answers 是列表，取第一个
                if self.dataset_name == "squad_v2":
                    if isinstance(ground_truth, list):
                        ground_truth = ground_truth[0] if ground_truth else None

                if not question or not ground_truth or not documents:
                    continue

                formatted_item = {
                    "question": str(question).strip(),
                    "ground_truth": ground_truth,
                    "golden_docs": documents,  # 黄金参考文档
                    "metadata": {
                        "dataset": self.dataset_name,
                        "sample_id": idx,
                    },
                }

                formatted_data.append(formatted_item)

            except Exception as e:
                print(f"[DatasetLoader] Error processing sample {idx}: {e}")
                continue

        return formatted_data

    def _post_process_crud_rag(self, item: dict) -> tuple:
        """CRUD-RAG 数据集后处理。

        数据结构：event（事件）, news1/news2/news3（参考新闻）,
        questions（可能包含多个问题）, answers（对应答案）, thoughts
        """
        question = str(item.get("questions", "")).strip()
        ground_truth = str(item.get("answers", "")).strip()

        if not question:
            return None, None, None

        # 合并三篇新闻作为参考文档
        documents = []
        for field in ["news1", "news2", "news3"]:
            news_text = item.get(field, "")
            if news_text:
                documents.append(str(news_text))

        # 如果有 event 描述，也加入文档
        event = item.get("event", "")
        if event:
            documents.insert(0, str(event))

        return question, ground_truth, documents

    def _extract_field(self, item: dict, field_path: str) -> Any:
        """从嵌套字典中提取字段（支持点分隔路径，如 answers.text）。"""
        if field_path in item:
            return item[field_path]

        parts = field_path.split(".")
        value = item
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None

        return value

    def _extract_documents(self, item: dict, doc_field: str) -> list[str]:
        """提取文档列表，兼容多种数据格式。"""
        docs = self._extract_field(item, doc_field)

        if not docs:
            return []

        if isinstance(docs, str):
            return [docs]

        if isinstance(docs, list):
            formatted_docs = []
            for doc in docs:
                if isinstance(doc, str):
                    formatted_docs.append(doc)
                elif isinstance(doc, list):
                    # HotpotQA 格式：[[title, text], ...]
                    if len(doc) >= 2 and isinstance(doc[1], str):
                        formatted_docs.append(f"{doc[0]}: {doc[1]}")
                    else:
                        formatted_docs.extend([str(d) for d in doc if d])
                elif isinstance(doc, dict):
                    for field in ["text", "content", "passage"]:
                        if field in doc:
                            formatted_docs.append(str(doc[field]))
                            break
            return formatted_docs

        return []

    def get_dataset_info(self) -> dict[str, Any]:
        """获取数据集信息。"""
        config = self._config_mapping[self.dataset_name]
        return {
            "name": self.dataset_name,
            "hf_path": self.SUPPORTED_DATASETS[self.dataset_name],
            "language": self.language,
            "subset_size": self.subset_size,
            "description": config.get("description", ""),
            "fields": list(config["field_mapping"].keys()),
        }

    def extract_corpus(self, dataset: list[dict[str, Any]]) -> list[str]:
        """从测试集提取全量去重语料库。

        将所有样本的 golden_docs 合并去重，作为向量化的检索源。
        评估时从语料库中检索，与黄金参考文档对比计算检索指标。

        Args:
            dataset: 从 load() 返回的测试集

        Returns:
            去重后的文档列表
        """
        seen = set()
        corpus = []

        for sample in dataset:
            # 兼容新旧字段名
            docs = sample.get("golden_docs") or sample.get("documents") or []
            for doc in docs:
                if not doc:
                    continue
                # 用文本前200字符做去重 key（避免全文比较的开销）
                doc_key = doc.strip()[:200]
                if doc_key not in seen:
                    seen.add(doc_key)
                    corpus.append(doc.strip())

        print(f"[DatasetLoader] Extracted corpus: {len(corpus)} unique documents "
              f"(from {len(dataset)} samples)")
        return corpus


class LocalDatasetLoader:
    """本地测试集加载器（备用方案）。

    支持 JSONL 格式，每行一个 JSON 对象，需包含 question / ground_truth / documents 字段。
    """

    def __init__(self, jsonl_path: str | None = None):
        self.jsonl_path = jsonl_path

    def load(self) -> list[dict[str, Any]]:
        """从本地文件加载测试集。"""
        import json

        if not self.jsonl_path or not Path(self.jsonl_path).exists():
            return self._get_sample_data()

        data = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if all(key in item for key in ["question", "ground_truth", "documents"]):
                        data.append(item)
                except json.JSONDecodeError:
                    continue

        return data

    def _get_sample_data(self) -> list[dict[str, Any]]:
        """返回示例数据（用于快速验证评估流程）。"""
        return [
            {
                "question": "什么是知识库助手？",
                "ground_truth": "知识库助手是基于RAG技术的问答系统，可以检索知识库并生成答案。",
                "documents": [
                    "知识库助手是一个智能问答系统，结合了信息检索和文本生成技术。",
                    "RAG（Retrieval-Augmented Generation）是检索增强生成的缩写。",
                    "知识库助手可以回答用户关于知识库中存储的信息的问题。",
                ],
                "metadata": {"dataset": "sample", "sample_id": 0},
            },
            {
                "question": "如何构建知识库？",
                "ground_truth": "通过加载文档、文本分割、向量化和存储到向量数据库来构建知识库。",
                "documents": [
                    "构建知识库需要先收集相关文档，包括PDF、Word、Markdown等格式。",
                    "文本分割将长文档切分成适合检索的片段。",
                    "向量化使用嵌入模型将文本转换为向量表示。",
                    "向量数据库存储向量并支持相似度检索。",
                ],
                "metadata": {"dataset": "sample", "sample_id": 1},
            },
        ]


def load_dataset_by_name(
    dataset_name: Literal["superclue_c3", "crud_rag", "hotpotqa", "squad_v2", "amnesty_qa"],
    subset_size: int | None = None,
    **kwargs,
) -> list[dict[str, Any]]:
    """快速加载测试集。

    Args:
        dataset_name: 测试集名称
        subset_size: 样本数量
        **kwargs: 其他参数传递给 DatasetLoader

    Returns:
        QA 对列表
    """
    loader = DatasetLoader(dataset_name, subset_size, **kwargs)
    return loader.load()
