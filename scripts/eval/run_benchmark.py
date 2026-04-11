"""RAGAS 基准评估执行器。

规范化的 6 步评估流程：
  Step 1: 加载公开测试集，提取 question / ground_truth / golden_docs / corpus
  Step 2: 将全量语料库向量化，导入专用 Qdrant 集合（隔离原有知识库）
  Step 3: 使用测试问题通过 RAG 系统进行检索与答案生成
  Step 4: 以黄金参考文档为基准，使用 RAGAS 计算检索阶段指标
  Step 5: 结合标准答案评估生成阶段与端到端指标
  Step 6: 输出完整评估报告与 bad case 分析

使用示例：
    # 完整评估
    python scripts/eval/run_benchmark.py --dataset crud_rag --subset-size 30

    # 只评估生成质量（跳过检索，直接用黄金文档作为上下文）
    python scripts/eval/run_benchmark.py --dataset crud_rag --skip-ingest

    # 只导入语料库
    python scripts/eval/run_benchmark.py --dataset crud_rag --ingest-only

    # 清理评估集合
    python scripts/eval/run_benchmark.py --dataset crud_rag --cleanup
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from core.evaluation.dataset_loader import DatasetLoader
from core.evaluation.eval_ingestor import EvalDataIngestor
from core.evaluation.evaluator import RagasEvaluator
from core.providers.model_provider import ModelRegistry
from core.types import SearchMode
from webapp.bootstrap import get_settings

from qdrant_client import QdrantClient


# ============================================================
#  辅助函数
# ============================================================

def search_eval_collection(
    client: QdrantClient, collection_name: str,
    query_vector: list[float], top_k: int = 10,
) -> list[dict[str, Any]]:
    """从评估集合中检索 top_k 相关文档。"""
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
    )
    return [
        {"content": hit.payload.get("content", ""), "score": hit.score}
        for hit in hits
    ]


def query_llm(chat_provider, question: str, contexts: list[str]) -> str:
    """用检索到的上下文调用 LLM 生成答案。"""
    context_text = "\n\n".join(f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts))
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个知识库助手。请根据提供的参考资料回答用户问题。"
                "如果参考资料中没有相关信息，请说明。不要编造信息。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"参考资料：\n{context_text}\n\n"
                f"用户问题：{question}\n\n"
                f"请根据参考资料回答上述问题："
            ),
        },
    ]
    return chat_provider.generate(messages)


# ============================================================
#  6 步评估流程
# ============================================================

def run_evaluation(
    dataset: list[dict[str, Any]],
    settings: Any,
    registry: ModelRegistry,
    embedding_provider: Any,
    chat_provider: Any,
    dataset_name: str,
    top_k: int = 10,
    skip_ingest: bool = False,
    batch_size: int = 5,
    loader: DatasetLoader | None = None,
    retrieval_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行规范化的 6 步评估流程。"""

    # ================================================================
    # Step 1: 加载测试集（已在调用前完成），提取全量语料库
    # ================================================================
    print(f"\n{'='*70}")
    print("Step 1/6: 准备测试数据")
    print(f"{'='*70}")
    print(f"  测试样本数:     {len(dataset)}")

    corpus = loader.extract_corpus(dataset) if (loader and not skip_ingest) else []
    if not skip_ingest:
        print(f"  全量语料库:     {len(corpus)} 篇去重文档（作为检索源）")
        print(f"  黄金参考文档:   每条样本独立的标注相关文档（用于评估检索质量）")

    # ================================================================
    # Step 2: 全量语料库向量化并导入专用向量数据库
    # ================================================================
    eval_collection = f"{dataset_name}_eval"
    ingest_stats = {}

    if skip_ingest:
        print(f"\n{'='*70}")
        print("Step 2/6: [跳过] --skip-ingest 模式，直接用黄金文档作为上下文")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print("Step 2/6: 语料库向量化，导入专用 Qdrant 集合")
        print(f"{'='*70}")

        ingestor = EvalDataIngestor(
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key,
            embedding_provider=embedding_provider,
            text_dim=registry.embeddings["text"].get("dimensions", 1024),
        )
        ingest_stats = ingestor.ingest_corpus(corpus, dataset_name)
        if "error" in ingest_stats:
            return {"error": ingest_stats["error"]}
        print(f"  导入完成: {ingest_stats}")

    # ================================================================
    # Step 3: 使用测试问题检索 + LLM 生成答案
    # ================================================================
    print(f"\n{'='*70}")
    print("Step 3/6: 检索 + 答案生成")
    print(f"{'='*70}")

    qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)

    questions, answers, contexts = [], [], []
    ground_truths, golden_docs_list = [], []
    failed_samples = []
    stats = {
        "total_samples": len(dataset),
        "successful_samples": 0,
        "failed_samples": 0,
        "retrieval_latencies": [],
        "generation_latencies": [],
    }
    start_time = time.time()

    for idx, sample in enumerate(dataset):
        try:
            pct = (idx + 1) / len(dataset) * 100
            print(f"\r  进度: {idx+1}/{len(dataset)} ({pct:.1f}%)", end="", flush=True)

            question = sample["question"]
            gt = sample["ground_truth"]
            golden = sample.get("golden_docs") or sample.get("documents") or []

            if skip_ingest:
                # 跳过检索：直接用黄金参考文档作为上下文
                retrieved = golden
                ret_latency = 0.0
            else:
                # 从专用向量库检索
                t0 = time.time()
                query_vec = embedding_provider.embed_query(question)
                hits = search_eval_collection(qdrant_client, eval_collection, query_vec, top_k)
                ret_latency = (time.time() - t0) * 1000
                retrieved = [h["content"] for h in hits if h["content"]]

            if not retrieved:
                failed_samples.append({"sample_id": idx, "question": question, "error": "No contexts"})
                stats["failed_samples"] += 1
                continue

            # LLM 生成答案
            t0 = time.time()
            answer = query_llm(chat_provider, question, retrieved)
            gen_latency = (time.time() - t0) * 1000

            questions.append(question)
            answers.append(answer)
            contexts.append(retrieved)
            ground_truths.append(gt)
            golden_docs_list.append(golden)
            stats["successful_samples"] += 1
            stats["retrieval_latencies"].append(ret_latency)
            stats["generation_latencies"].append(gen_latency)

        except Exception as e:
            failed_samples.append({"sample_id": idx, "question": sample.get("question", ""), "error": str(e)})
            stats["failed_samples"] += 1

    print()
    stats["total_latency_ms"] = (time.time() - start_time) * 1000
    print(f"  成功: {stats['successful_samples']}, 失败: {stats['failed_samples']}")

    if not questions:
        return {"error": "No successful evaluations", "failed_samples": failed_samples, "stats": stats}

    # ================================================================
    # Step 4: 以黄金参考文档为基准，计算检索阶段指标
    # ================================================================
    print(f"\n{'='*70}")
    print("Step 4/6: 检索阶段评估（以黄金参考文档为基准）")
    print(f"{'='*70}")
    print("  指标: recall@k, precision@k, MRR, coverage@k")

    # ================================================================
    # Step 5: 结合标准答案，评估生成阶段与端到端指标
    # ================================================================
    print(f"\n{'='*70}")
    print("Step 5/6: 生成阶段 + 端到端评估（以标准答案为基准）")
    print(f"{'='*70}")
    print("  指标: faithfulness, answer_relevancy, answer_correctness, answer_similarity, ...")

    evaluator = RagasEvaluator(llm=chat_provider, embeddings=embedding_provider)

    try:
        # golden_docs_list 作为 documents 传入，用于计算 mrr@k / coverage@k 等检索指标
        # ground_truths 作为标准答案，用于计算 answer_correctness / answer_similarity 等生成指标
        eval_results = evaluator.evaluate_with_ground_truth(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths,
            documents=golden_docs_list if not skip_ingest else None,
            batch_size=batch_size,
        )
    except Exception as e:
        print(f"  RAGAS 评估失败: {e}")
        traceback.print_exc()
        return {"error": f"RAGAS failed: {e}", "failed_samples": failed_samples, "stats": stats}

    if "error" in eval_results:
        return {"error": eval_results["error"], "failed_samples": failed_samples, "stats": stats}

    # ================================================================
    # Step 6: 输出完整评估报告与 bad case 分析
    # ================================================================
    print(f"\n{'='*70}")
    print("Step 6/6: 生成报告 + bad case 分析")
    print(f"{'='*70}")

    report = build_report(
        eval_results, stats, dataset_name, top_k, skip_ingest,
        eval_collection, failed_samples, ingest_stats,
        retrieval_config=retrieval_config,
    )

    return report


def build_report(
    eval_results: dict, stats: dict, dataset_name: str,
    top_k: int, skip_ingest: bool, eval_collection: str,
    failed_samples: list, ingest_stats: dict,
    retrieval_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建完整报告，包含 bad case 分析。"""

    summary_extra: dict[str, Any] = {}
    if retrieval_config:
        summary_extra["retrieval_config"] = retrieval_config

    report = {
        "summary": {
            **eval_results["summary"],
            "dataset": dataset_name,
            "top_k": top_k,
            "skip_ingest": skip_ingest,
            "eval_collection": eval_collection if not skip_ingest else None,
            "ingest_stats": ingest_stats,
            "stats": stats,
            **summary_extra,
        },
        "metrics": eval_results["metrics"],
        "samples": eval_results["samples"],
        "failed_samples": failed_samples,
        "bad_cases": [],
    }

    # 性能统计
    if stats["retrieval_latencies"]:
        report["summary"]["avg_retrieval_latency_ms"] = sum(stats["retrieval_latencies"]) / len(stats["retrieval_latencies"])
    if stats["generation_latencies"]:
        report["summary"]["avg_generation_latency_ms"] = sum(stats["generation_latencies"]) / len(stats["generation_latencies"])

    # ── Bad case 分析 ──
    metrics = eval_results.get("metrics", {})
    threshold = 0.5

    for sample in eval_results.get("samples", []):
        sample_metrics = sample.get("metrics", {})
        # 检测异常指标
        weak_metrics = {}
        for metric_name, score in sample_metrics.items():
            if isinstance(score, (int, float)) and score < threshold:
                weak_metrics[metric_name] = score

        if weak_metrics:
            report["bad_cases"].append({
                "sample_id": sample.get("sample_id"),
                "question": sample.get("question", "")[:200],
                "answer": sample.get("answer", "")[:200],
                "ground_truth": str(sample.get("ground_truth", ""))[:200],
                "weak_metrics": weak_metrics,
                "analysis": _analyze_bad_case(weak_metrics),
            })

    # 按 weak_metrics 数量排序（问题最严重的排前面）
    report["bad_cases"].sort(key=lambda x: len(x["weak_metrics"]), reverse=True)

    return report


def _analyze_bad_case(weak_metrics: dict[str, float]) -> str:
    """对 bad case 进行简单原因分析。"""
    reasons = []
    if "faithfulness" in weak_metrics:
        reasons.append("答案可能包含幻觉（faithfulness过低），未忠实于检索上下文")
    if "recall@k" in weak_metrics:
        reasons.append("检索召回率低，黄金参考文档中关键信息未被检索到")
    if "precision@k" in weak_metrics:
        reasons.append("检索精度低，检索结果中包含大量不相关内容")
    if "answer_relevancy" in weak_metrics:
        reasons.append("答案与问题相关性低，可能偏题")
    if "answer_correctness" in weak_metrics:
        reasons.append("答案正确性低，与标准答案偏差较大")
    if "answer_similarity" in weak_metrics:
        reasons.append("答案语义与标准答案相似度低")
    if "mrr@k" in weak_metrics:
        reasons.append("相关文档排名靠后，首位文档不相关")
    if "coverage@k" in weak_metrics:
        reasons.append("上下文覆盖率低，检索内容未能有效覆盖参考文档")
    if "answer_coverage" in weak_metrics:
        reasons.append("答案覆盖率不足，缺少标准答案中的关键信息")
    return "; ".join(reasons) if reasons else "多个指标低于阈值"


# ============================================================
#  报告输出
# ============================================================

def save_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告已保存: {output_path}")


def print_summary(report: dict[str, Any]) -> None:
    # 处理 error 报告
    if "error" in report:
        print(f"\n{'='*80}")
        print("RAGAS BENCHMARK EVALUATION - ERROR")
        print(f"{'='*80}")
        print(f"\n  Error: {report['error']}")
        stats = report.get("stats", {})
        if stats:
            print(f"  Total samples:    {stats.get('total_samples', 0)}")
            print(f"  Successful:       {stats.get('successful_samples', 0)}")
            print(f"  Failed:           {stats.get('failed_samples', 0)}")
        failed = report.get("failed_samples", [])
        if failed:
            print(f"\n  Failed samples ({len(failed)}):")
            for f_item in failed[:5]:
                print(f"    #{f_item['sample_id']}: {f_item['error'][:80]}")
        print(f"\n{'='*80}")
        return

    summary = report.get("summary", {})
    metrics = report.get("metrics", {})
    bad_cases = report.get("bad_cases", [])
    failed = report.get("failed_samples", [])

    print(f"\n{'='*80}")
    print("RAGAS BENCHMARK EVALUATION REPORT")
    print(f"{'='*80}")

    print(f"\n  Dataset:          {summary.get('dataset', 'N/A')}")
    print(f"  Total samples:    {summary.get('total_samples', 0)}")
    print(f"  Successful:       {summary.get('stats', {}).get('successful_samples', 0)}")
    print(f"  Failed:           {summary.get('stats', {}).get('failed_samples', 0)}")
    print(f"  Top K:            {summary.get('top_k', 'N/A')}")
    print(f"  Skip ingest:      {summary.get('skip_ingest', False)}")
    print(f"  Eval collection:  {summary.get('eval_collection', 'N/A')}")

    if "ingest_stats" in summary and summary["ingest_stats"]:
        ing = summary["ingest_stats"]
        print(f"  Corpus docs:      {ing.get('total_documents', 'N/A')}")
        print(f"  Corpus chunks:    {ing.get('total_chunks', 'N/A')}")

    if "avg_retrieval_latency_ms" in summary:
        print(f"  Avg retrieval:    {summary['avg_retrieval_latency_ms']:.2f}ms")
    if "avg_generation_latency_ms" in summary:
        print(f"  Avg generation:   {summary['avg_generation_latency_ms']:.2f}ms")

    # 检索配置
    retrieval_cfg = summary.get("retrieval_config", {})
    if retrieval_cfg:
        print(f"\n  Retrieval Config:")
        print(f"    Search mode:      {retrieval_cfg.get('search_mode', 'N/A')}")
        print(f"    Reranker:         {retrieval_cfg.get('reranker_provider', 'none')}"
              f" (enabled={retrieval_cfg.get('reranker_enabled', False)})")
        if retrieval_cfg.get("reranker_model"):
            print(f"    Reranker model:   {retrieval_cfg['reranker_model']}")

    # 指标（按阶段分组）
    if metrics:
        print(f"\n  Metrics:")
        retrieval = ["recall@k", "precision@k", "mrr@k", "coverage@k"]
        generation = ["faithfulness", "answer_relevancy", "answer_coverage"]
        e2e = ["answer_correctness", "answer_similarity"]

        for label, keys in [("Retrieval", retrieval), ("Generation", generation), ("End-to-End", e2e)]:
            label_metrics = {k: v for k, v in metrics.items() if k in keys}
            if label_metrics:
                print(f"\n    [{label}]")
                for name, vals in label_metrics.items():
                    mean = vals.get("mean", 0)
                    std = vals.get("std", 0)
                    print(f"      {name:.<35} {mean:.4f} (±{std:.4f})")

    # Bad cases
    if bad_cases:
        print(f"\n  Bad Cases ({len(bad_cases)} / {summary.get('total_samples', 0)}):")
        for i, bc in enumerate(bad_cases[:5]):
            q = bc["question"][:50] + "..." if len(bc["question"]) > 50 else bc["question"]
            print(f"    #{i+1} [{len(bc['weak_metrics'])} weak] {q}")
            print(f"       {bc['analysis']}")
        if len(bad_cases) > 5:
            print(f"    ... and {len(bad_cases) - 5} more (see report)")

    # Failed
    if failed:
        print(f"\n  Failed samples ({len(failed)}):")
        for f_item in failed[:3]:
            print(f"    #{f_item['sample_id']}: {f_item['error'][:60]}")

    print(f"\n{'='*80}")


def load_config(config_path: Path) -> dict[str, Any]:
    import yaml
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ============================================================
#  主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RAGAS 基准评估（6步标准化流程）")

    parser.add_argument("--dataset", type=str,
        choices=["superclue_c3", "crud_rag", "hotpotqa", "squad_v2", "amnesty_qa"],
        default="crud_rag", help="测试集名称")
    parser.add_argument("--subset-size", type=int, default=None, help="评估样本数量")
    parser.add_argument("--language", type=str, choices=["zh", "en"], default="zh")
    parser.add_argument("--top-k", type=int, default=10, help="检索 top_k")
    parser.add_argument("--search-mode", type=str,
        choices=["hybrid", "text_only", "multimodal", "graph_first"],
        default="hybrid", help="检索模式（hybrid/text_only/multimodal/graph_first）")
    parser.add_argument("--reranker", type=str,
        choices=["none", "llm", "cross-encoder"],
        default=None, help="重排方法（none/llm/cross-encoder，默认从配置文件读取）")
    parser.add_argument("--skip-ingest", action="store_true", help="跳过导入，只评估生成质量")
    parser.add_argument("--ingest-only", action="store_true", help="只导入语料库")
    parser.add_argument("--cleanup", action="store_true", help="清理评估集合")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=5, help="RAGAS 批处理大小")

    args = parser.parse_args()

    if args.config and args.config.exists():
        config = load_config(args.config)
        if "dataset" in config:
            args.dataset = config["dataset"].get("name", args.dataset)
            args.subset_size = config["dataset"].get("subset_size", args.subset_size)
            args.language = config["dataset"].get("language", args.language)

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(f"runtime/eval/benchmark_{args.dataset}_{timestamp}.json")

    try:
        settings = get_settings()
        registry = ModelRegistry(settings.model_registry_path)
        embedding_provider = registry.build_embedding()

        # 仅清理
        if args.cleanup:
            ingestor = EvalDataIngestor(
                qdrant_url=settings.qdrant_url, qdrant_api_key=settings.qdrant_api_key,
                embedding_provider=embedding_provider,
                text_dim=registry.embeddings["text"].get("dimensions", 1024),
            )
            ingestor.cleanup_collection(args.dataset)
            return

        # 仅导入语料库
        if args.ingest_only:
            loader = DatasetLoader(dataset_name=args.dataset, subset_size=args.subset_size, language=args.language)
            dataset = loader.load()
            if not dataset:
                print("[Error] Failed to load dataset"); sys.exit(1)
            corpus = loader.extract_corpus(dataset)
            ingestor = EvalDataIngestor(
                qdrant_url=settings.qdrant_url, qdrant_api_key=settings.qdrant_api_key,
                embedding_provider=embedding_provider,
                text_dim=registry.embeddings["text"].get("dimensions", 1024),
            )
            result = ingestor.ingest_corpus(corpus, args.dataset)
            print(f"[Done] {result}")
            return

        # 构建检索配置信息（写入报告，用于追踪实验参数）
        reranker_cfg = registry._cfg.get("reranker", {})
        reranker_enabled = reranker_cfg.get("enabled", False)

        # --reranker 命令行参数优先于配置文件
        if args.reranker is not None:
            if args.reranker == "none":
                reranker_enabled = False
                reranker_provider = "none"
            else:
                reranker_enabled = True
                reranker_provider = args.reranker
        else:
            reranker_provider = reranker_cfg.get("provider", "none") if reranker_enabled else "none"

        retrieval_config = {
            "search_mode": args.search_mode,
            "reranker_enabled": reranker_enabled,
            "reranker_provider": reranker_provider,
            "reranker_model": reranker_cfg.get("model", "") if reranker_enabled else "",
        }

        # 完整评估
        chat_provider = registry.build_chat("fast_model")

        loader = DatasetLoader(dataset_name=args.dataset, subset_size=args.subset_size, language=args.language)
        dataset = loader.load()
        if not dataset:
            print("[Error] Failed to load dataset"); sys.exit(1)

        report = run_evaluation(
            dataset=dataset,
            settings=settings,
            registry=registry,
            embedding_provider=embedding_provider,
            chat_provider=chat_provider,
            dataset_name=args.dataset,
            top_k=args.top_k,
            skip_ingest=args.skip_ingest,
            batch_size=args.batch_size,
            loader=loader,
            retrieval_config=retrieval_config,
        )

        save_report(report, args.output)
        print_summary(report)

        if not args.skip_ingest:
            print(f"\n  清理命令: python scripts/eval/run_benchmark.py --dataset {args.dataset} --cleanup")

    except Exception as e:
        print(f"[Error] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
