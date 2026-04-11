"""离线批量评估脚本。

功能：
1. 加载 QA 日志（runtime/eval/qa_logs.jsonl）
2. 调用 RagasEvaluator 计算指标
3. 生成评估报告（JSON 格式）
4. 支持筛选条件（按时间、search_mode）

使用示例：
    # 评估所有日志
    python scripts/eval/run_batch_eval.py

    # 评估最近 100 条
    python scripts/eval/run_batch_eval.py --last-n 100

    # 只评估 hybrid 模式
    python scripts/eval/run_batch_eval.py --mode hybrid

    # 评估并生成详细报告
    python scripts/eval/run_batch_eval.py --output runtime/eval/eval_report_$(date +%Y%m%d).json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from core.evaluation.evaluator import RagasEvaluator
from webapp.bootstrap import get_embeddings, get_registry


def load_qa_logs(log_file: Path, mode: str | None = None, last_n: int | None = None) -> list[dict]:
    """加载 QA 日志。

    Args:
        log_file: JSONL 日志文件路径
        mode: 筛选特定的 search_mode（如 'hybrid', 'text_only'）
        last_n: 只加载最近 n 条记录

    Returns:
        QA 日志列表
    """
    if not log_file.exists():
        print(f"[ERROR] Log file not found: {log_file}")
        return []

    logs = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                log = json.loads(line)

                # 按 mode 筛选
                if mode and log.get("search_mode") != mode:
                    continue

                logs.append(log)
            except json.JSONDecodeError as e:
                print(f"[WARNING] Skipping malformed log line: {e}")
                continue

    # 只取最近 n 条
    if last_n:
        logs = logs[-last_n:]

    print(f"[INFO] Loaded {len(logs)} QA logs from {log_file}")
    return logs


def save_report(report: dict, output_file: Path) -> None:
    """保存评估报告。"""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Evaluation report saved to: {output_file}")


def print_summary(report: dict) -> None:
    """打印评估摘要。"""
    print("\n" + "=" * 80)
    print("RAGAS EVALUATION REPORT")
    print("=" * 80)

    summary = report.get("summary", {})
    print(f"\n📊 Summary:")
    print(f"   Total samples: {summary.get('total_samples', 0)}")
    print(f"   Valid samples: {summary.get('valid_samples', 0)}")
    print(f"   Evaluated at: {summary.get('evaluated_at', '')}")

    metrics = report.get("metrics", {})
    if metrics:
        print(f"\n📈 Metrics:")
        for metric_name, values in metrics.items():
            mean = values.get("mean", 0)
            min_val = values.get("min", 0)
            max_val = values.get("max", 0)
            print(f"   {metric_name:.<30} {mean:.4f} (min: {min_val:.4f}, max: {max_val:.4f})")
    else:
        print("\n⚠️  No metrics calculated")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="RAG 系统批量评估脚本")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=ROOT_DIR / "runtime" / "eval" / "qa_logs.jsonl",
        help="QA 日志文件路径 (default: runtime/eval/qa_logs.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="评估报告输出路径 (default: runtime/eval/eval_report_YYYYmmdd_HHMMSS.json)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hybrid", "text_only", "multimodal", "graph_first"],
        default=None,
        help="筛选特定的 search_mode",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=None,
        help="只评估最近 n 条记录",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="单条评估模式（用于测试）",
    )

    args = parser.parse_args()

    # 加载 QA 日志
    qa_logs = load_qa_logs(args.log_file, mode=args.mode, last_n=args.last_n)

    if not qa_logs:
        print("[ERROR] No QA logs found. Please run the system first to generate logs.")
        sys.exit(1)

    # 初始化评估器
    try:
        print("[INFO] Initializing evaluator...")
        registry = get_registry()
        embeddings = get_embeddings()
        evaluator = RagasEvaluator(llm=registry.build_chat("fast_model"), embeddings=embeddings)
    except Exception as e:
        print(f"[ERROR] Failed to initialize evaluator: {e}")
        sys.exit(1)

    # 执行评估
    if args.single:
        # 单条评估（用于测试）
        print("[INFO] Running single evaluation (first log only)...")
        log = qa_logs[0]
        retrieval_result = log.get("retrieval_result", {})
        items = retrieval_result.get("items", [])

        contexts = []
        for item in items:
            content = item.get("content") or item.get("text") or item.get("chunk_text") or ""
            if content:
                contexts.append(content)

        if not contexts:
            print("[ERROR] No contexts found in first log")
            sys.exit(1)

        single_result = evaluator.evaluate_single(
            query=log.get("rewritten_query") or log.get("query", ""),
            answer=log.get("answer", ""),
            contexts=contexts,
        )

        print("\n" + "=" * 80)
        print("SINGLE EVALUATION RESULT")
        print("=" * 80)
        print(json.dumps(single_result, ensure_ascii=False, indent=2))
        sys.exit(0)

    print(f"[INFO] Evaluating {len(qa_logs)} samples...")
    report = evaluator.evaluate_batch(qa_logs)

    # 读取 reranker 配置
    try:
        from core.config.settings import get_settings as _get_settings
        from core.providers.model_provider import ModelRegistry as _MR
        _settings = _get_settings()
        _registry = _MR(_settings.model_registry_path)
        _reranker_cfg = _registry._cfg.get("reranker", {})
        reranker_info = {
            "enabled": _reranker_cfg.get("enabled", False),
            "provider": _reranker_cfg.get("provider", "none") if _reranker_cfg.get("enabled", False) else "none",
            "model": _reranker_cfg.get("model", ""),
        }
    except Exception:
        reranker_info = {"enabled": False, "provider": "unknown", "model": ""}

    # 添加元信息
    report["summary"]["evaluated_at"] = datetime.now().isoformat()
    report["summary"]["log_file"] = str(args.log_file)
    report["config"] = {
        "mode": args.mode,
        "last_n": args.last_n,
        "reranker": reranker_info,
    }

    # 打印摘要
    print_summary(report)

    # 保存报告
    if args.output is None:
        # 生成默认文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = ROOT_DIR / "runtime" / "eval" / f"eval_report_{timestamp}.json"

    save_report(report, args.output)

    # 打印采样结果（前3条）
    if "samples" in report and report["samples"]:
        print(f"\n📝 Sample Results (first 3):")
        for i, sample in enumerate(report["samples"][:3]):
            print(f"\n   Sample {i + 1} (trace_id: {sample.get('trace_id', 'N/A')[:8]}...):")
            print(f"   Query: {sample.get('query', '')[:60]}...")
            if "metrics" in sample:
                for metric, value in sample["metrics"].items():
                    print(f"     {metric:.<25} {value:.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
