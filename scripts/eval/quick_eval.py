"""快速评估脚本 - 一键运行中文/英文测试集评估。

使用方法：
    # 评估中文CRUD-RAG（100条样本，hybrid模式）
    python scripts/eval/quick_eval.py --zh

    # 评估英文HotpotQA（50条样本，hybrid模式）
    python scripts/eval/quick_eval.py --en

    # 自定义样本数量和输出目录
    python scripts/eval/quick_eval.py --zh --subset-size 50 --output-dir runtime/eval/custom

    # 只评估检索阶段
    python scripts/eval/quick_eval.py --zh --retrieval-only

    # 指定检索模式
    python scripts/eval/quick_eval.py --zh --search-mode text_only

    # 列出所有支持的测试集
    python scripts/eval/quick_eval.py --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))


class QuickEvaluator:
    """快速评估执行器。"""

    # 预定义配置
    PRESETS = {
        "zh": {
            "name": "中文CRUD-RAG评估",
            "dataset": "crud_rag",
            "subset_size": 100,
            "language": "zh",
            "search_mode": "hybrid",
            "description": "基于事件的中文问答测试集（3199条），含事件描述和三篇参考新闻",
        },
        "zh_small": {
            "name": "中文CRUD-RAG（小样本）",
            "dataset": "crud_rag",
            "subset_size": 30,
            "language": "zh",
            "search_mode": "hybrid",
            "description": "快速测试，30条样本",
        },
        "zh_superclue": {
            "name": "中文SuperCLUE-C3阅读理解",
            "dataset": "superclue_c3",
            "subset_size": 50,
            "language": "zh",
            "search_mode": "hybrid",
            "description": "中文阅读理解测试集（4792条），instruction=问题, input=短文, output=答案",
        },
        "zh_amnesty": {
            "name": "RAGAS官方示例（快速验证）",
            "dataset": "amnesty_qa",
            "subset_size": 20,
            "language": "en",
            "search_mode": "hybrid",
            "description": "RAGAS官方示例数据集，仅20条，用于快速验证评估流程是否正常",
        },
        "en": {
            "name": "英文HotpotQA",
            "dataset": "hotpotqa",
            "subset_size": 50,
            "language": "en",
            "search_mode": "hybrid",
            "description": "英文多跳问答数据集，context 格式为 [[title, text], ...]",
        },
        "en_small": {
            "name": "英文HotpotQA（小样本）",
            "dataset": "hotpotqa",
            "subset_size": 20,
            "language": "en",
            "search_mode": "hybrid",
            "description": "快速测试，20条样本",
        },
        "en_squad": {
            "name": "英文SQuAD v2",
            "dataset": "squad_v2",
            "subset_size": 100,
            "language": "en",
            "search_mode": "hybrid",
            "description": "Stanford 抽取式问答 v2，含不可回答的问题",
        },
    }

    def __init__(self):
        self.output_dir = Path("runtime/eval")

    def list_presets(self):
        """列出所有预定义配置。"""
        print("可用的评估配置：\n")
        for key, config in self.PRESETS.items():
            print(f"  {key:20s} - {config['name']}")
            print(f"{'':22s}   {config['description']}")
            print(f"{'':22s}   数据集: {config['dataset']}, 样本数: {config['subset_size']}\n")

    def run_evaluation(
        self,
        preset: str | None = None,
        dataset: str | None = None,
        subset_size: int | None = None,
        language: str = "zh",
        search_mode: str = "hybrid",
        skip_ingest: bool = False,
        ingest_only: bool = False,
        cleanup: bool = False,
        output_dir: str | None = None,
    ):
        """运行评估。

        Args:
            preset: 预定义配置名称
            dataset: 数据集名称
            subset_size: 样本数量
            language: 语言
            search_mode: 检索模式
            skip_ingest: 跳过文档导入，直接用测试集文档评估生成质量
            ingest_only: 只导入测试文档，不评估
            cleanup: 清理评估集合
            output_dir: 输出目录
        """
        # 使用预定义配置或自定义参数
        if preset:
            if preset not in self.PRESETS:
                print(f"[Error] 未知的预设配置: {preset}")
                print("\n请使用 --list 查看所有可用配置")
                return False

            config = self.PRESETS[preset]
            dataset = config["dataset"]
            subset_size = config["subset_size"]
            language = config["language"]
            search_mode = config["search_mode"]
            eval_name = config["name"]
        else:
            if not dataset:
                print("[Error] 必须指定 --dataset 或 --preset")
                return False
            eval_name = f"自定义评估 ({dataset})"

        # 设置输出目录
        if output_dir:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 生成输出文件名（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{dataset}_{timestamp}.json"

        # 构建命令
        cmd = [
            sys.executable,
            str(ROOT_DIR / "scripts" / "eval" / "run_benchmark.py"),
            "--dataset", dataset,
            "--language", language,
            "--search-mode", search_mode,
            "--top-k", "10",
            "--output", str(output_file),
        ]

        if subset_size:
            cmd.extend(["--subset-size", str(subset_size)])

        if skip_ingest:
            cmd.append("--skip-ingest")

        if ingest_only:
            cmd.append("--ingest-only")

        if cleanup:
            cmd.append("--cleanup")

        # 打印评估信息
        print("=" * 80)
        print("RAGAS 快速评估")
        print("=" * 80)
        print(f"\n评估名称: {eval_name}")
        print(f"数据集: {dataset}")
        print(f"样本数量: {subset_size or '全部'}")
        print(f"语言: {language}")
        print(f"检索模式: {search_mode}")
        print(f"跳过导入: {skip_ingest}")
        print(f"仅导入: {ingest_only}")
        print(f"清理集合: {cleanup}")
        print(f"输出文件: {output_file}")
        print("\n" + "-" * 80 + "\n")

        # 执行命令
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print("\n" + "=" * 80)
            print("✅ 评估完成！")
            print("=" * 80)
            print(f"\n报告文件: {output_file}")

            # 检查是否生成了CSV
            csv_file = output_file.with_suffix(".csv")
            if csv_file.exists():
                print(f"CSV文件: {csv_file}")

            # 检查是否生成了异常报告
            anomaly_file = self.output_dir / f"{dataset}_{timestamp}_anomalies.csv"
            if anomaly_file.exists():
                print(f"异常报告: {anomaly_file}")

            print("\n下一步操作:")
            print(f"  1. 查看JSON报告: {output_file}")
            print(f"  2. 导入CSV到Excel分析")
            print(f"  3. 运行: python -m json.tool {output_file} | less")

            return True

        except subprocess.CalledProcessError as e:
            print(f"\n❌ 评估失败: {e}")
            return False

    def run_all_zh(self, **kwargs):
        """运行所有中文测试集评估。"""
        print("运行中文测试集评估...\n")
        results = {}

        for preset in ["zh", "zh_superclue"]:
            print(f"\n{'=' * 80}")
            print(f"运行 {preset}")
            print(f"{'=' * 80}\n")
            success = self.run_evaluation(preset=preset, **kwargs)
            results[preset] = success

        print("\n" + "=" * 80)
        print("中文测试集评估完成")
        print("=" * 80)
        for preset, success in results.items():
            status = "✅" if success else "❌"
            print(f"{status} {preset}")

        return all(results.values())

    def run_all_en(self, **kwargs):
        """运行所有英文测试集评估。"""
        print("运行英文测试集评估...\n")
        results = {}

        for preset in ["en", "en_squad"]:
            print(f"\n{'=' * 80}")
            print(f"运行 {preset}")
            print(f"{'=' * 80}\n")
            success = self.run_evaluation(preset=preset, **kwargs)
            results[preset] = success

        print("\n" + "=" * 80)
        print("英文测试集评估完成")
        print("=" * 80)
        for preset, success in results.items():
            status = "✅" if success else "❌"
            print(f"{status} {preset}")

        return all(results.values())


def main():
    parser = argparse.ArgumentParser(description="RAGAS 快速评估脚本")

    # 评估类型
    parser.add_argument(
        "--zh",
        action="store_true",
        help="评估中文CRUD-RAG测试集（100条）",
    )
    parser.add_argument(
        "--zh-small",
        action="store_true",
        help="评估中文CRUD-RAG小样本（30条）",
    )
    parser.add_argument(
        "--zh-superclue",
        action="store_true",
        help="评估中文SuperCLUE-C3测试集（50条）",
    )
    parser.add_argument(
        "--zh-amnesty",
        action="store_true",
        help="RAGAS官方示例（20条，快速验证）",
    )
    parser.add_argument(
        "--en",
        action="store_true",
        help="评估英文HotpotQA测试集（50条）",
    )
    parser.add_argument(
        "--en-small",
        action="store_true",
        help="评估英文HotpotQA小样本（20条）",
    )
    parser.add_argument(
        "--en-nq",
        action="store_true",
        help="评估英文SQuAD v2测试集（100条）",
    )
    parser.add_argument(
        "--all-zh",
        action="store_true",
        help="运行所有中文测试集",
    )
    parser.add_argument(
        "--all-en",
        action="store_true",
        help="运行所有英文测试集",
    )

    # 通用参数
    parser.add_argument(
        "--preset",
        type=str,
        help="使用预定义配置（使用--list查看所有配置）",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="自定义数据集名称（superclue_c3, crud_rag, hotpotqa, squad_v2, amnesty_qa）",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        help="样本数量",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["zh", "en"],
        help="语言",
    )
    parser.add_argument(
        "--search-mode",
        type=str,
        choices=["hybrid", "text_only", "multimodal", "graph_first"],
        help="检索模式",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="[已废弃] 请使用 --skip-ingest 代替",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="跳过文档导入，直接用测试集文档评估生成质量（不评估检索）",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="只导入测试文档到Qdrant，不执行评估",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="清理评估集合",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="输出目录（默认: runtime/eval）",
    )

    # 工具选项
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有预定义配置",
    )

    args = parser.parse_args()

    evaluator = QuickEvaluator()

    # 列出配置
    if args.list:
        evaluator.list_presets()
        return

    # 执行评估
    kwargs = {
        "output_dir": args.output_dir,
        "skip_ingest": args.skip_ingest or args.retrieval_only,
        "ingest_only": args.ingest_only,
        "cleanup": args.cleanup,
    }

    if args.search_mode:
        kwargs["search_mode"] = args.search_mode

    if args.preset:
        success = evaluator.run_evaluation(preset=args.preset, **kwargs)
        sys.exit(0 if success else 1)

    # 根据参数运行评估
    if args.zh:
        success = evaluator.run_evaluation(preset="zh", **kwargs)
    elif args.zh_small:
        success = evaluator.run_evaluation(preset="zh_small", **kwargs)
    elif args.zh_superclue:
        success = evaluator.run_evaluation(preset="zh_superclue", **kwargs)
    elif args.zh_amnesty:
        success = evaluator.run_evaluation(preset="zh_amnesty", **kwargs)
    elif args.en:
        success = evaluator.run_evaluation(preset="en", **kwargs)
    elif args.en_small:
        success = evaluator.run_evaluation(preset="en_small", **kwargs)
    elif args.en_nq:
        success = evaluator.run_evaluation(preset="en_squad", **kwargs)
    elif args.all_zh:
        success = evaluator.run_all_zh(**kwargs)
    elif args.all_en:
        success = evaluator.run_all_en(**kwargs)
    elif args.dataset:
        success = evaluator.run_evaluation(
            dataset=args.dataset,
            subset_size=args.subset_size,
            language=args.language or "zh",
            **kwargs,
        )
    else:
        print("[Error] 请指定评估类型（使用 --zh, --en, --preset, --dataset 或 --list）")
        parser.print_help()
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
