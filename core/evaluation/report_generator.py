"""评估报告生成器。

支持JSON和CSV格式的评估报告导出，自动标注异常样本（低分样本）。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class ReportGenerator:
    """评估报告生成器。"""

    def __init__(self, report_data: dict[str, Any]):
        """初始化报告生成器。

        Args:
            report_data: 评估报告数据（从RagasEvaluator返回）
        """
        self.report_data = report_data

    def save_json(self, output_path: Path, pretty: bool = True) -> None:
        """保存为JSON格式。

        Args:
            output_path: 输出文件路径
            pretty: 是否格式化输出
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(self.report_data, f, ensure_ascii=False, indent=2)
            else:
                json.dump(self.report_data, f, ensure_ascii=False)

        print(f"[ReportGenerator] JSON report saved to: {output_path}")

    def save_csv(self, output_path: Path, include_contexts: bool = False) -> None:
        """保存为CSV格式（便于Excel分析）。

        Args:
            output_path: 输出文件路径
            include_contexts: 是否包含上下文文本（会增加文件大小）
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 准备CSV数据
        csv_data = []

        samples = self.report_data.get("samples", [])
        for sample in samples:
            row = {
                "sample_id": sample.get("sample_id"),
                "question": sample.get("question", ""),
                "answer": sample.get("answer", ""),
                "ground_truth": str(sample.get("ground_truth", "")),
            }

            # 添加指标分数
            metrics = sample.get("metrics", {})
            for metric_name, score in metrics.items():
                row[metric_name] = score

            # 添加上下文（可选）
            if include_contexts:
                contexts = sample.get("contexts", [])
                row["contexts"] = "\n\n---\n\n".join(contexts[:3])  # 只取前3个上下文
                row["num_contexts"] = len(contexts)

            csv_data.append(row)

        if not csv_data:
            print("[ReportGenerator] No samples to export to CSV")
            return

        # 创建DataFrame并保存
        df = pd.DataFrame(csv_data)

        # 确保指标列存在
        metric_names = [
            "faithfulness",
            "answer_relevancy",
            "precision@k",
            "recall@k",
            "entity_recall@k",
            "answer_correctness",
            "answer_similarity",
            "mrr@k",
            "answer_coverage",
            "coverage@k",
        ]

        for metric in metric_names:
            if metric not in df.columns:
                df[metric] = None

        # 按样本ID排序
        if "sample_id" in df.columns:
            df = df.sort_values("sample_id")

        # 保存CSV
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"[ReportGenerator] CSV report saved to: {output_path}")
        print(f"[ReportGenerator] Total samples: {len(df)}")

    def generate_summary_csv(self, output_path: Path) -> None:
        """生成指标摘要CSV（按指标统计）。

        Args:
            output_path: 输出文件路径
        """
        metrics = self.report_data.get("metrics", {})
        if not metrics:
            print("[ReportGenerator] No metrics to summarize")
            return

        summary_data = []
        for metric_name, stats in metrics.items():
            summary_data.append(
                {
                    "metric": metric_name,
                    "mean": stats.get("mean"),
                    "std": stats.get("std"),
                    "min": stats.get("min"),
                    "max": stats.get("max"),
                    "median": stats.get("median"),  # 如果需要可以计算
                }
            )

        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"[ReportGenerator] Summary CSV saved to: {output_path}")

    def annotate_anomalies(
        self, threshold: float = 0.5, metrics: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """标注异常样本（低分样本）。

        Args:
            threshold: 异常阈值（分数低于此值视为异常）
            metrics: 需要检查的指标列表（None表示检查所有指标）

        Returns:
            异常样本列表
        """
        if metrics is None:
            # 默认检查关键指标
            metrics = [
                "faithfulness",
                "answer_correctness",
                "answer_similarity",
                "recall@k",
            ]

        anomalies = []
        samples = self.report_data.get("samples", [])

        for sample in samples:
            sample_metrics = sample.get("metrics", {})
            anomaly_metrics = {}

            for metric in metrics:
                if metric in sample_metrics:
                    score = sample_metrics[metric]
                    if score < threshold:
                        anomaly_metrics[metric] = score

            if anomaly_metrics:
                anomalies.append(
                    {
                        "sample_id": sample.get("sample_id"),
                        "question": sample.get("question", "")[:100],  # 截断显示
                        "anomaly_metrics": anomaly_metrics,
                        "all_metrics": sample_metrics,
                    }
                )

        return anomalies

    def save_anomaly_report(self, anomalies: list[dict[str, Any]], output_path: Path) -> None:
        """保存异常样本报告。

        Args:
            anomalies: 异常样本列表（从annotate_anomalies返回）
            output_path: 输出文件路径
        """
        if not anomalies:
            print("[ReportGenerator] No anomalies to report")
            return

        anomaly_data = []
        for anomaly in anomalies:
            row = {
                "sample_id": anomaly["sample_id"],
                "question": anomaly["question"],
            }

            # 添加异常指标
            for metric, score in anomaly["anomaly_metrics"].items():
                row[f"{metric}_score"] = score
                row[f"{metric}_anomaly"] = True

            # 添加所有指标
            for metric, score in anomaly["all_metrics"].items():
                if metric not in row:
                    row[metric] = score

            anomaly_data.append(row)

        df = pd.DataFrame(anomaly_data)

        # 确保列的顺序
        columns = ["sample_id", "question"]
        metric_columns = [col for col in df.columns if col not in columns]
        df = df[columns + sorted(metric_columns)]

        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"[ReportGenerator] Anomaly report saved to: {output_path}")
        print(f"[ReportGenerator] Total anomalies: {len(anomalies)}")

    def print_summary(self) -> None:
        """打印报告摘要。"""
        summary = self.report_data.get("summary", {})
        metrics = self.report_data.get("metrics", {})

        print("\n" + "=" * 80)
        print("EVALUATION REPORT SUMMARY")
        print("=" * 80)

        print(f"\n📊 Dataset:")
        print(f"   Total samples: {summary.get('total_samples', 0)}")
        print(f"   Valid samples: {summary.get('valid_samples', 0)}")
        if "search_mode" in summary:
            print(f"   Search mode: {summary['search_mode']}")
        if "top_k" in summary:
            print(f"   Top K: {summary['top_k']}")

        print(f"\n📈 Metrics:")
        for metric_name, stats in metrics.items():
            mean = stats.get("mean", 0)
            std = stats.get("std", 0)
            print(f"   {metric_name:.<35} {mean:.4f} (±{std:.4f})")

        # 异常样本数量
        anomalies = self.annotate_anomalies()
        if anomalies:
            print(f"\n⚠️  Anomalies detected: {len(anomalies)} samples")

        print("\n" + "=" * 80)

    def export_all(
        self,
        base_path: Path,
        include_contexts: bool = False,
        anomaly_threshold: float = 0.5,
    ) -> None:
        """导出所有报告格式。

        Args:
            base_path: 基础输出路径（不含扩展名）
            include_contexts: 是否包含上下文文本
            anomaly_threshold: 异常检测阈值
        """
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # JSON报告
        json_path = base_path.with_suffix(".json")
        self.save_json(json_path)

        # CSV报告
        csv_path = base_path.with_suffix(".csv")
        self.save_csv(csv_path, include_contexts=include_contexts)

        # 摘要CSV
        summary_csv_path = base_path.parent / f"{base_path.stem}_summary.csv"
        self.generate_summary_csv(summary_csv_path)

        # 异常报告
        anomalies = self.annotate_anomalies(threshold=anomaly_threshold)
        if anomalies:
            anomaly_path = base_path.parent / f"{base_path.stem}_anomalies.csv"
            self.save_anomaly_report(anomalies, anomaly_path)

        print(f"\n[ReportGenerator] All reports exported to: {base_path.parent}")


# 快速导出函数
def export_report(
    report_data: dict[str, Any],
    output_path: Path,
    include_contexts: bool = False,
    anomaly_threshold: float = 0.5,
) -> None:
    """快速导出评估报告。

    Args:
        report_data: 评估报告数据
        output_path: 输出路径（不含扩展名）
        include_contexts: 是否包含上下文
        anomaly_threshold: 异常检测阈值
    """
    generator = ReportGenerator(report_data)
    generator.export_all(
        base_path=output_path,
        include_contexts=include_contexts,
        anomaly_threshold=anomaly_threshold,
    )


def convert_json_to_csv(json_path: Path, csv_path: Path | None = None) -> None:
    """将JSON报告转换为CSV格式。

    Args:
        json_path: JSON报告路径
        csv_path: CSV输出路径（None表示自动生成）
    """
    if not json_path.exists():
        print(f"[Error] JSON file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    if csv_path is None:
        csv_path = json_path.with_suffix(".csv")

    generator = ReportGenerator(report_data)
    generator.save_csv(csv_path, include_contexts=True)
