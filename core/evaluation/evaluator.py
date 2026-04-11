"""Ragas 评估器封装。

职责：
1. 封装 Ragas 无监督指标（无需 golden answers）
2. 支持 LLM-as-a-Judge 自动评估
3. 批量评估和增量评估
4. 生成结构化评估报告

支持的指标（无需测试集）：
- faithfulness: 答案忠实度（检测幻觉）
- answer_relevancy: 答案相关性
- precision@k: 上下文精确率（需要 LLM 判断相关性）
- recall@k: 上下文召回率（需要 LLM 判断相关性）
- entity_recall@k: 实体召回率

需要测试集的指标（预留接口）：
- answer_correctness: 答案正确性
- answer_similarity: 答案相似度
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

# Ragas 指标
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,  # 需要ground_truth
    answer_relevancy,
    answer_similarity,  # 需要ground_truth
    context_entity_recall,
    context_precision,
    context_recall,
    faithfulness,
)

# 注意：Ragas 0.2.x 的 API 可能变化，这里使用适配器模式
from datasets import Dataset
import numpy as np

# RAGAS 内部指标名 → 标准评估指标名
_RAGAS_METRIC_NAME_MAP = {
    "context_recall": "recall@k",
    "context_precision": "precision@k",
    "context_entity_recall": "entity_recall@k",
}


class RagasEvaluator:
    """Ragas 评估器（支持无监督和基于ground_truth的评估）。

    支持的指标：
    检索阶段：
    - recall@k: 上下文召回率
    - precision@k: 上下文精度
    - mrr@k: 平均倒数排名（自定义实现）

    生成阶段：
    - faithfulness: 忠实度（检测幻觉）
    - answer_relevancy: 答案相关性
    - answer_coverage: 答案覆盖率（基于ground_truth）

    端到端阶段：
    - answer_correctness: 答案正确性（需要ground_truth）
    - answer_similarity: 答案相似度（需要ground_truth）
    - coverage@k: 上下文覆盖率（自定义实现）
    """

    def __init__(self, llm: Any, embeddings: Any):
        """
        Args:
            llm: LangChain LLM 实例，用于 LLM-as-a-Judge
            embeddings: 嵌入模型，用于 answer_relevancy 和相似度计算
        """
        self.llm = llm
        self.embeddings = embeddings
        self._ragas_llm = None
        self._ragas_embeddings = None

        # 配置 RAGAS 使用的 LLM 和 embeddings
        self._configure_ragas()

    def _configure_ragas(self) -> None:
        """配置 Ragas 使用指定的 LLM 和 embeddings。

        使用 langchain_community 的 ChatZhipuAI（智谱官方适配），
        避免 langchain_openai.ChatOpenAI 与智谱 API 的参数不兼容问题。
        """
        try:
            # ── 从项目的 ChatProvider 提取 API Key ──
            llm_client = getattr(self.llm, "client", None)
            ragas_model = getattr(self.llm, "model", "glm-4-flash")

            api_key = ""
            if llm_client:
                api_key = llm_client.api_key or ""

            if not api_key:
                # 尝试从环境变量读取
                import os
                api_key = os.environ.get("ZHIPUAI_API_KEY", "")

            if not api_key:
                print("[RagasEvaluator] Warning: No API key found. "
                      "Set ZHIPUAI_API_KEY env var.")

            # ── 使用 ChatZhipuAI（智谱官方 LangChain 适配器） ──
            from langchain_community.chat_models import ChatZhipuAI

            self._ragas_llm = ChatZhipuAI(
                model=ragas_model,
                api_key=api_key if api_key else None,
                temperature=0.1,
                max_tokens=2048,
            )

            # ── 配置 Embeddings：包装项目的 EmbeddingProvider 为 Langchain 接口 ──
            from langchain_core.embeddings import Embeddings

            class _ProviderEmbeddings(Embeddings):
                """将项目的 EmbeddingProvider 包装为 Langchain Embeddings 接口。"""

                def __init__(self, provider: Any):
                    self._provider = provider

                def embed_documents(self, texts: list[str]) -> list[list[float]]:
                    return self._provider.embed_text(texts)

                def embed_query(self, text: str) -> list[float]:
                    result = self._provider.embed_text([text])
                    return result[0] if result else []

            self._ragas_embeddings = _ProviderEmbeddings(self.embeddings)

            print(f"[RagasEvaluator] Configured RAGAS with ChatZhipuAI model={ragas_model}, "
                  f"embed_dims={getattr(self.embeddings, 'text_dimensions', 'N/A')}")

        except Exception as e:
            print(f"[RagasEvaluator] Warning: Failed to configure Ragas: {e}")
            traceback.print_exc()

    def evaluate_batch(self, qa_logs: list[dict[str, Any]]) -> dict[str, Any]:
        """批量评估 QA 日志。

        Args:
            qa_logs: 从 EvaluationRecorder 读取的日志列表

        Returns:
            评估结果字典，包含各项指标和统计信息
        """
        if not qa_logs:
            return {"error": "No QA logs provided"}

        # 转换为 Ragas Dataset 格式
        dataset = self._convert_to_ragas_dataset(qa_logs)

        if len(dataset) == 0:
            return {"error": "No valid QA pairs for evaluation"}

        # 配置指标列表（无监督指标，无需 golden answers）
        metrics = [
            faithfulness,  # 忠实度：答案是否基于上下文
            answer_relevancy,  # 答案相关性
            context_precision,  # 上下文精确率
            context_recall,  # 上下文召回率
            context_entity_recall,  # 实体召回率
        ]

        # 执行评估
        try:
            eval_kwargs = {
                "dataset": dataset,
                "metrics": metrics,
                "batch_size": 5,
            }
            if self._ragas_llm:
                eval_kwargs["llm"] = self._ragas_llm
            if self._ragas_embeddings:
                eval_kwargs["embeddings"] = self._ragas_embeddings

            results = evaluate(**eval_kwargs)

            # 转换为字典
            return self._parse_results(results, qa_logs)
        except Exception as e:
            traceback.print_exc()
            return {"error": f"Evaluation failed: {str(e)}"}

    def _convert_to_ragas_dataset(self, qa_logs: list[dict[str, Any]]) -> Dataset:
        """将 QA 日志转换为 Ragas Dataset 格式。

        Ragas 要求的数据格式：
        {
            "question": ["问题1", "问题2", ...],
            "answer": ["答案1", "答案2", ...],
            "contexts": [["上下文1-1", "上下文1-2"], ["上下文2-1"], ...]
        }
        """
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
        }

        for log in qa_logs:
            try:
                # 提取问题（优先使用 rewritten_query）
                question = log.get("rewritten_query") or log.get("query", "")
                if not question:
                    continue

                # 提取答案
                answer = log.get("answer", "")
                if not answer:
                    continue

                # 提取上下文（检索结果中的 content）
                retrieval_result = log.get("retrieval_result", {})
                items = retrieval_result.get("items", [])

                contexts = []
                for item in items:
                    # 从 item 中提取 content
                    content = (
                        item.get("content")
                        or item.get("text")
                        or item.get("chunk_text")
                        or ""
                    )
                    if content:
                        contexts.append(content)

                # 如果上下为空，跳过（Ragas 需要上下文）
                if not contexts:
                    continue

                # 添加到数据集
                data["question"].append(question)
                data["answer"].append(answer)
                data["contexts"].append(contexts)

            except Exception as e:
                # 单条记录转换失败不影响整体
                print(f"[RagasEvaluator] Skipping malformed log: {e}")
                continue

        return Dataset.from_dict(data)

    def _parse_results(self, results: Any, qa_logs: list[dict[str, Any]]) -> dict[str, Any]:
        """解析 Ragas 评估结果。

        Args:
            results: Ragas 评估结果对象
            qa_logs: 原始日志，用于补充元信息

        Returns:
            结构化评估报告
        """
        # 转换为字典（兼容不同版本的 RAGAS）
        if hasattr(results, '_scores_dict'):
            results_dict = results._scores_dict
        elif hasattr(results, 'scores') and isinstance(results.scores, list):
            results_dict = {k: [d[k] for d in results.scores] for k in results.scores[0].keys()} if results.scores else {}
        else:
            results_dict = dict(results)

        # 计算统计信息
        report = {
            "summary": {
                "total_samples": len(qa_logs),
                "valid_samples": len(results_dict.get("question", [])),
                "evaluated_at": "",  # TODO: 添加时间戳
            },
            "metrics": {},
            "samples": [],
        }

        # 计算每个指标的均值
        metric_names = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "context_entity_recall",
        ]

        for ragas_name in metric_names:
            if ragas_name in results_dict:
                values = results_dict[ragas_name]
                if values:
                    display_name = _RAGAS_METRIC_NAME_MAP.get(ragas_name, ragas_name)
                    report["metrics"][display_name] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "values": values,  # 保存原始值用于后续分析
                    }

        # 构建每条样本的详细结果
        for idx, log in enumerate(qa_logs):
            if idx >= len(results_dict.get("question", [])):
                break

            sample = {
                "trace_id": log.get("trace_id"),
                "query": log.get("query"),
                "rewritten_query": log.get("rewritten_query"),
                "answer": log.get("answer"),
                "latency_ms": log.get("latency_ms"),
                "search_mode": log.get("search_mode"),
                "metrics": {},
            }

            # 添加每个指标的分数
            for ragas_name in metric_names:
                if ragas_name in results_dict and idx < len(results_dict[ragas_name]):
                    display_name = _RAGAS_METRIC_NAME_MAP.get(ragas_name, ragas_name)
                    sample["metrics"][display_name] = results_dict[ragas_name][idx]

            report["samples"].append(sample)

        return report

    def evaluate_single(
        self, query: str, answer: str, contexts: list[str]
    ) -> dict[str, float]:
        """评估单个问答对。

        Args:
            query: 用户问题
            answer: 生成的答案
            contexts: 检索上下文列表

        Returns:
            各项指标分数
        """
        if not contexts:
            return {"error": "No contexts provided"}

        dataset = Dataset.from_dict(
            {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
            }
        )

        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            context_entity_recall,
        ]

        try:
            eval_kwargs = {
                "dataset": dataset,
                "metrics": metrics,
            }
            if self._ragas_llm:
                eval_kwargs["llm"] = self._ragas_llm
            if self._ragas_embeddings:
                eval_kwargs["embeddings"] = self._ragas_embeddings

            results = evaluate(**eval_kwargs)

            # 兼容不同版本的 RAGAS
            if hasattr(results, '_scores_dict'):
                results_dict = results._scores_dict
            elif hasattr(results, 'scores') and isinstance(results.scores, list):
                results_dict = {k: [d[k] for d in results.scores] for k in results.scores[0].keys()} if results.scores else {}
            else:
                results_dict = dict(results)

            return {
                "faithfulness": results_dict.get("faithfulness", [0])[0],
                "answer_relevancy": results_dict.get("answer_relevancy", [0])[0],
                "precision@k": results_dict.get("context_precision", [0])[0],
                "recall@k": results_dict.get("context_recall", [0])[0],
                "entity_recall@k": results_dict.get("context_entity_recall", [0])[0],
            }
        except Exception as e:
            return {"error": str(e)}

    # ==================== 基于 ground_truth 的评估（公开测试集） ====================

    def evaluate_with_ground_truth(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str | list[str]],
        documents: list[list[str]] | None = None,
        batch_size: int = 5,
    ) -> dict[str, Any]:
        """使用 ground_truth 评估（支持公开测试集）。

        Args:
            questions: 问题列表
            answers: 生成的答案列表
            contexts: 检索到的上下文列表（每个元素是上下文字符串列表）
            ground_truths: 标准答案列表
            documents: 参考文档列表（用于计算 mrr@k 和 coverage@k）
            batch_size: 批处理大小

        Returns:
            包含所有指标的评估结果
        """
        if not all([questions, answers, contexts, ground_truths]):
            return {"error": "Missing required fields"}

        if len(questions) != len(answers) or len(questions) != len(ground_truths):
            return {"error": "Input lists must have the same length"}

        # 构建RAGAS数据集格式
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }

        # 添加ground_truth到数据集
        dataset = Dataset.from_dict(dataset_dict)

        # 配置指标列表
        metrics = [
            faithfulness,  # 忠实度
            answer_relevancy,  # 答案相关性
            context_precision,  # 上下文精度
            context_recall,  # 上下文召回率
            context_entity_recall,  # 实体召回率
            answer_correctness,  # 答案正确性（需要ground_truth）
            answer_similarity,  # 答案相似度（需要ground_truth）
        ]

        # 执行RAGAS评估
        try:
            eval_kwargs = {
                "dataset": dataset,
                "metrics": metrics,
                "batch_size": batch_size,
            }
            if self._ragas_llm:
                eval_kwargs["llm"] = self._ragas_llm
            if self._ragas_embeddings:
                eval_kwargs["embeddings"] = self._ragas_embeddings

            results = evaluate(**eval_kwargs)
            # RAGAS 0.2.x: EvaluationResult 没有 to_dict()，用 _scores_dict 获取结果
            results_dict = {}
            if hasattr(results, '_scores_dict'):
                results_dict = results._scores_dict
            elif hasattr(results, 'scores') and isinstance(results.scores, list):
                results_dict = {k: [d[k] for d in results.scores] for k in results.scores[0].keys()} if results.scores else {}
            else:
                results_dict = dict(results)
        except Exception as e:
            traceback.print_exc()
            return {"error": f"RAGAS evaluation failed: {str(e)}"}

        # 计算自定义指标
        custom_metrics = {}

        # 计算 MRR（如果有documents）
        if documents:
            mrr_scores = self._calculate_mrr(contexts, documents)
            custom_metrics["mrr@k"] = mrr_scores

        # 计算答案覆盖率
        completeness_scores = self._calculate_answer_completeness(answers, ground_truths)
        custom_metrics["answer_coverage"] = completeness_scores

        # 计算上下文覆盖率
        if documents:
            utilization_scores = self._calculate_context_utilization(contexts, documents)
            custom_metrics["coverage@k"] = utilization_scores

        # 构建评估报告
        return self._parse_ground_truth_results(
            results_dict, custom_metrics, questions, answers, contexts, ground_truths
        )

    def _calculate_mrr(self, contexts: list[list[str]], documents: list[list[str]]) -> list[float]:
        """计算平均倒数排名（Mean Reciprocal Rank）。"""
        mrr_scores = []

        for ctx_list, doc_list in zip(contexts, documents):
            if not ctx_list or not doc_list:
                mrr_scores.append(0.0)
                continue

            doc_set = set()
            for doc in doc_list:
                doc_key = doc.strip()[:200]
                doc_set.add(doc_key)

            reciprocal_rank = 0.0
            for rank, context in enumerate(ctx_list, 1):
                ctx_key = context.strip()[:200]
                if ctx_key in doc_set:
                    reciprocal_rank = 1.0 / rank
                    break

            mrr_scores.append(reciprocal_rank)

        return mrr_scores

    def _calculate_answer_completeness(
        self, answers: list[str], ground_truths: list[str | list[str]]
    ) -> list[float]:
        """计算答案完整性（基于ground_truth的关键词覆盖率）。"""
        completeness_scores = []

        for answer, ground_truth in zip(answers, ground_truths):
            if not answer or not ground_truth:
                completeness_scores.append(0.0)
                continue

            if isinstance(ground_truth, list):
                gt_text = " ".join(ground_truth)
            else:
                gt_text = ground_truth

            answer_words = set(answer.lower().split())
            gt_words = set(gt_text.lower().split())

            if not gt_words:
                completeness_scores.append(0.0)
                continue

            coverage = len(answer_words & gt_words) / len(gt_words)
            completeness_scores.append(min(coverage, 1.0))

        return completeness_scores

    def _calculate_context_utilization(
        self, contexts: list[list[str]], documents: list[list[str]]
    ) -> list[float]:
        """计算上下文利用率（检索内容覆盖参考文档的程度）。"""
        utilization_scores = []

        for ctx_list, doc_list in zip(contexts, documents):
            if not ctx_list or not doc_list:
                utilization_scores.append(0.0)
                continue

            all_docs_text = " ".join(doc_list)
            doc_words = set(all_docs_text.lower().split())

            if not doc_words:
                utilization_scores.append(0.0)
                continue

            covered_words = set()
            for context in ctx_list:
                ctx_words = set(context.lower().split())
                covered_words.update(ctx_words & doc_words)

            utilization = len(covered_words) / len(doc_words)
            utilization_scores.append(min(utilization, 1.0))

        return utilization_scores

    def _parse_ground_truth_results(
        self,
        ragas_results: dict[str, Any],
        custom_metrics: dict[str, list[float]],
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str | list[str]],
    ) -> dict[str, Any]:
        """解析包含ground_truth的评估结果，构建结构化报告。"""
        from datetime import datetime

        report = {
            "summary": {
                "total_samples": len(questions),
                "evaluated_at": datetime.now().isoformat(),
                "has_ground_truth": True,
            },
            "metrics": {},
            "samples": [],
        }

        # 处理RAGAS指标
        ragas_metric_names = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "context_entity_recall",
            "answer_correctness",
            "answer_similarity",
        ]

        for ragas_name in ragas_metric_names:
            if ragas_name in ragas_results:
                values = ragas_results[ragas_name]
                if values and len(values) > 0:
                    # 过滤 NaN/None 值
                    valid_values = [float(v) for v in values if v is not None and str(v) != 'nan']
                    if valid_values:
                        display_name = _RAGAS_METRIC_NAME_MAP.get(ragas_name, ragas_name)
                        report["metrics"][display_name] = {
                            "mean": float(np.mean(valid_values)),
                            "std": float(np.std(valid_values)),
                            "min": float(np.min(valid_values)),
                            "max": float(np.max(valid_values)),
                            "values": valid_values,
                        }

        # 处理自定义指标
        for metric_name, values in custom_metrics.items():
            if values and len(values) > 0:
                report["metrics"][metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values": [float(v) for v in values],
                }

        # 构建样本详情
        for idx in range(len(questions)):
            sample = {
                "sample_id": idx,
                "question": questions[idx],
                "answer": answers[idx],
                "ground_truth": ground_truths[idx],
                "contexts": contexts[idx],
                "metrics": {},
            }

            for ragas_name in ragas_metric_names:
                display_name = _RAGAS_METRIC_NAME_MAP.get(ragas_name, ragas_name)
                if display_name in report["metrics"] and idx < len(report["metrics"][display_name]["values"]):
                    sample["metrics"][display_name] = report["metrics"][display_name]["values"][idx]

            for metric_name, values in custom_metrics.items():
                if metric_name in report["metrics"] and idx < len(values):
                    sample["metrics"][metric_name] = values[idx]

            report["samples"].append(sample)

        return report
