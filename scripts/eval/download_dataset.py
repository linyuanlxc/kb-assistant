"""下载公开测试集到本地。

将 Hugging Face 数据集下载并保存为 JSONL 文件，后续评估直接从本地读取，
无需重复访问 Hugging Face。

使用方法：
    # 下载中文CRUD-RAG（30条样本）
    python scripts/eval/download_dataset.py --dataset crud_rag --subset-size 30

    # 下载全量中文SuperCLUE-C3
    python scripts/eval/download_dataset.py --dataset superclue_c3

    # 下载英文HotpotQA（50条）
    python scripts/eval/download_dataset.py --dataset hotpotqa --subset-size 50

    # 下载所有推荐测试集
    python scripts/eval/download_dataset.py --all

    # 指定保存目录
    python scripts/eval/download_dataset.py --dataset crud_rag --output-dir data/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from core.evaluation.dataset_loader import DatasetLoader

DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "eval"


def download_and_save(
    dataset_name: str,
    subset_size: int | None = None,
    language: str = "zh",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> bool:
    """下载数据集并保存为 JSONL。"""
    print(f"\n{'='*60}")
    print(f"Downloading: {dataset_name}")
    print(f"{'='*60}")

    try:
        loader = DatasetLoader(
            dataset_name=dataset_name,
            subset_size=subset_size,
            language=language,
        )

        dataset = loader.load()

        if not dataset:
            print(f"[Error] No valid data loaded for {dataset_name}")
            return False

        # 保存为 JSONL
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{dataset_name}.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"\n[Done] Saved {len(dataset)} samples to: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

        # 打印前2条样本预览
        print(f"\n  Preview (first 2 samples):")
        for i, item in enumerate(dataset[:2]):
            q = item["question"][:80]
            gt = str(item["ground_truth"])[:60]
            docs = len(item.get("documents", []))
            print(f"    [{i}] Q: {q}...")
            print(f"        A: {gt}...")
            print(f"        Docs: {docs} paragraphs")

        return True

    except Exception as e:
        print(f"[Error] Failed to download {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="下载公开测试集到本地")

    parser.add_argument("--dataset", type=str,
        choices=["superclue_c3", "crud_rag", "hotpotqa", "squad_v2", "amnesty_qa"],
        help="下载指定数据集")
    parser.add_argument("--subset-size", type=int, default=None,
        help="下载数量（不指定则全部）")
    parser.add_argument("--language", type=str, choices=["zh", "en"], default="zh")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="保存目录（默认: data/eval/）")
    parser.add_argument("--all", action="store_true",
        help="下载所有推荐测试集")

    args = parser.parse_args()

    if not args.all and not args.dataset:
        parser.print_help()
        print("\n请指定 --dataset 或 --all")
        sys.exit(1)

    if args.all:
        # 推荐的测试集及其默认参数
        datasets_to_download = [
            ("crud_rag", 100, "zh"),
            ("superclue_c3", 50, "zh"),
            ("hotpotqa", 50, "en"),
            ("amnesty_qa", None, "en"),
        ]
        results = {}
        for name, size, lang in datasets_to_download:
            ok = download_and_save(name, size, lang, args.output_dir)
            results[name] = ok

        print(f"\n{'='*60}")
        print("下载汇总")
        print(f"{'='*60}")
        for name, ok in results.items():
            print(f"  {'OK' if ok else 'FAIL':4s}  {name}")

        failed = [n for n, ok in results.items() if not ok]
        if failed:
            print(f"\n  失败: {', '.join(failed)}")
            sys.exit(1)
    else:
        ok = download_and_save(
            args.dataset, args.subset_size, args.language, args.output_dir
        )
        if not ok:
            sys.exit(1)

    print(f"\n所有文件保存在: {args.output_dir}")
    print("评估时使用: python scripts/eval/quick_eval.py --zh-small")


if __name__ == "__main__":
    main()
