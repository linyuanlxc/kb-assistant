"""用于构建 V2 知识库索引的命令行脚本。

用法：
    python scripts/build_kb.py
    python scripts/build_kb.py --full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from core.config.settings import load_settings
from core.indexing.builder import IndexBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 KB Assistant V2 的向量 / BM25 / 图索引")
    parser.add_argument("--full", action="store_true", help="执行全量重建，而不是增量更新")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    builder = IndexBuilder(settings)
    # 默认走增量构建，只有显式指定 --full 时才重建全部索引。
    summary = builder.build(incremental=not args.full)
    print("索引构建结果：")
    for k, v in summary.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
