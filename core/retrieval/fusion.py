"""混合检索的融合工具。

本模块包含：
- Reciprocal Rank Fusion（RRF）
- 动态权重归一化
- 跨 dense / sparse / graph / image 检索器的加权分数合并
"""

from __future__ import annotations

from collections import defaultdict

from core.types import RetrieverItem


def reciprocal_rank_fusion(rank_lists: list[list[RetrieverItem]], k: int = 60) -> dict[str, float]:
    """计算每个条目的 RRF 分数。

    Args:
        rank_lists: 来自多个独立检索器的排序列表。
        k: RRF 平滑常数，通常取 60 左右。
    """
    result = defaultdict(float)
    for items in rank_lists:
        for rank, item in enumerate(items, start=1):
            result[item.item_id] += 1.0 / (k + rank)
    return dict(result)


def normalize_weights(weights: dict[str, float], use_image: bool) -> dict[str, float]:
    """将检索权重归一化，使总和为 1。

    如果没有图片查询，则禁用 image 分支，并对其余分支重新归一化。
    """
    w = dict(weights)
    if not use_image:
        w["image_clip"] = 0.0
    total = sum(w.values())
    if total <= 0:
        return {"text_dense": 1.0, "bm25": 0.0, "graph": 0.0, "image_clip": 0.0}
    return {k: v / total for k, v in w.items()}


def weighted_merge(
    dense: list[RetrieverItem],
    bm25: list[RetrieverItem],
    graph: list[RetrieverItem],
    image: list[RetrieverItem],
    weights: dict[str, float],
    top_k: int,
) -> tuple[list[RetrieverItem], dict[str, float]]:
    """按配置权重合并多分支检索结果。"""
    id2item: dict[str, RetrieverItem] = {}
    merged_scores = defaultdict(float)

    def add(items: list[RetrieverItem], key: str) -> None:
        """累计每个候选项在当前分支上的分数贡献。"""
        for i in items:
            id2item[i.item_id] = i
            merged_scores[i.item_id] += i.score * weights.get(key, 0.0)

    add(dense, "text_dense")
    add(bm25, "bm25")
    add(graph, "graph")
    add(image, "image_clip")

    ranked = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for item_id, score in ranked:
        item = id2item[item_id]
        item.score = float(score)
        results.append(item)
    return results, dict(ranked)
