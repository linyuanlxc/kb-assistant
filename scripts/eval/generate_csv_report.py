"""从JSON评估报告生成CSV报告。

使用方法：
    # 从单个JSON文件生成CSV
    python scripts/eval/generate_csv_report.py --report runtime/eval/benchmark_*.json

    # 指定输出CSV路径
    python scripts/eval/generate_csv_report.py \
      --report runtime/eval/benchmark_*.json \
      --output runtime/eval/custom_name.csv

    # 包含上下文文本
    python scripts/eval/generate_csv_report.py \
      --report runtime/eval/benchmark_*.json \
      --include-contexts

    # 标注异常样本（分数低于阈值）
    python scripts/eval/generate_csv_report.py \
      --report runtime/eval/benchmark_*.json \
      --anomaly-threshold 0.6 \
      --generate-anomaly-report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from core.evaluation.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="从JSON评估报告生成CSV报告")

    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="JSON评估报告路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV输出路径（默认：与JSON同名，扩展名为.csv）",
    )
    parser.add_argument(
        "--include-contexts",
        action="store_true",
        help="包含上下文文本（会增加文件大小）",
    )
    parser.add_argument(
        "--generate-anomaly-report",
        action="store_true",
        help="生成异常样本报告",
    )
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=0.5,
        help="异常检测阈值（默认：0.5）",
    )
    parser.add_argument(
        "--generate-summary",
        action="store_true",
        help="生成指标摘要CSV",
    )

    args = parser.parse_args()

    # 检查输入文件
    if not args.report.exists():
        print(f"[Error] JSON报告文件不存在: {args.report}")
        sys.exit(1)

    # 确定输出路径
    if args.output is None:
        args.output = args.report.with_suffix(".csv")

    # 加载JSON报告
    import json

    with open(args.report, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    # 创建报告生成器
    generator = ReportGenerator(report_data)

    # 生成CSV报告
    print("=" * 80)
    print("生成CSV评估报告")
    print("=" * 80)
    print(f"\n输入JSON: {args.report}")
    print(f"输出CSV: {args.output}")
    print(f"包含上下文: {args.include_contexts}")
    print(f"异常阈值: {args.anomaly_threshold}")
    print(f"生成异常报告: {args.generate_anomaly_report}")
    print(f"生成摘要: {args.generate_summary}")

    print("\n" + "-" * 80 + "\n")

    # 保存主CSV报告
    generator.save_csv(args.output, include_contexts=args.include_contexts)

    # 生成摘要
    if args.generate_summary:
        summary_output = args.output.parent / f"{args.output.stem}_summary.csv"
        generator.generate_summary_csv(summary_output)

    # 生成异常报告
    if args.generate_anomaly_report:
        anomalies = generator.annotate_anomalies(threshold=args.anomaly_threshold)
        if anomalies:
            anomaly_output = args.output.parent / f"{args.output.stem}_anomalies.csv"
            generator.save_anomaly_report(anomalies, anomaly_output)
        else:
            print("[Info] 未发现异常样本")

    # 打印摘要
    generator.print_summary()

    print("\n" + "=" * 80)
    print("✅ CSV报告生成完成！")
    print("=" * 80)

    print(f"\nCSV文件: {args.output}")
    if args.generate_summary:
        print(f"摘要文件: {summary_output}")
    if args.generate_anomaly_report and anomalies:
        print(f"异常报告: {anomaly_output}")

    print("\n下一步操作:")
    print("  1. 用Excel或WPS打开CSV文件进行分析")
    print("  2. 筛选低分样本，分析失败原因")
    print("  3. 根据分析结果优化检索或生成策略")


if __name__ == "__main__":
    main()
