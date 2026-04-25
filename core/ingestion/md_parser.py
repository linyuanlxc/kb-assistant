"""Markdown 结构化解析器。

将 Markdown 文件解析为带层级的 JSON 结构，保留标题层级与内容归属关系。
输出格式示例：
[
    {"level": 1, "title": "第一章", "content": "...", "children": [...]},
    {"level": 2, "title": "第一节", "content": "...", "children": [...]},
]
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def parse_markdown(path: Path) -> list[dict[str, Any]]:
    """解析 Markdown 文件为结构化层级列表。

    Args:
        path: Markdown 文件路径。

    Returns:
        结构化节点列表，每个节点包含 level/title/content/children。
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=False)

    # 根节点（虚拟，用于挂载一级标题）
    root: dict[str, Any] = {"level": 0, "children": []}
    # 栈维护当前路径：[root, ..., current]
    stack: list[dict[str, Any]] = [root]
    # 当前节点累积的正文行
    pending_lines: list[str] = []

    def _flush_pending() -> None:
        """将累积的正文写入栈顶节点的 content。"""
        if pending_lines and stack:
            top = stack[-1]
            existing = top.get("content", "")
            extra = "\n".join(pending_lines)
            top["content"] = (existing + "\n" + extra).strip() if existing else extra
            pending_lines.clear()

    heading_re = re.compile(r"^(#{1,6})\s+(.+)$")

    for line in lines:
        m = heading_re.match(line)
        if m:
            # 遇到新标题，先将累积正文刷入当前节点
            _flush_pending()

            level = len(m.group(1))
            title = m.group(2).strip()

            new_node: dict[str, Any] = {
                "level": level,
                "title": title,
                "content": "",
                "children": [],
            }

            # 调整栈：弹出所有 level >= 当前 level 的节点
            while len(stack) > 1 and stack[-1]["level"] >= level:
                stack.pop()
            # 将新节点挂到栈顶节点的 children
            stack[-1]["children"].append(new_node)
            stack.append(new_node)
        else:
            # 非标题行，累积到 pending_lines
            if line.strip() or pending_lines:
                pending_lines.append(line)

    _flush_pending()
    return root["children"]


def structured_to_text(
    nodes: list[dict[str, Any]],
    include_headings: bool = True,
) -> str:
    """将结构化节点树展平为纯文本（保留标题标记）。

    用于兼容现有纯文本切分流程。
    """
    parts: list[str] = []

    def _walk(node: dict[str, Any]) -> None:
        if include_headings and node.get("title"):
            parts.append("#" * node["level"] + " " + node["title"])
        if node.get("content"):
            parts.append(node["content"])
        for child in node.get("children", []):
            _walk(child)

    for node in nodes:
        _walk(node)

    return "\n\n".join(parts)
