"""关键词搜索工具。

用于在原文中按关键词定位候选证据片段，返回命中行及上下文。
"""

from __future__ import annotations

from .base_tool import BaseTool, ToolResult


class KeywordSearch(BaseTool):
    """在文本中搜索关键词，返回命中行和上下文。"""

    name = "keyword_search"
    description = "按关键词搜索文本，返回命中行、行号和上下文，作为后续分析的证据候选。"

    def run(
        self,
        text: str,
        keywords: list[str] | None = None,
        context_lines: int = 1,
        max_results: int = 10,
    ) -> ToolResult:
        """执行关键词搜索。

        Args:
            text: 待搜索的全文。
            keywords: 关键词列表。
            context_lines: 每条命中前后保留的上下文行数。
            max_results: 最大返回命中数量。
        """
        if not text:
            return ToolResult(success=False, error="缺少 text，无法搜索关键词")

        normalized_keywords = _normalize_keywords(keywords or [])
        if not normalized_keywords:
            return ToolResult(success=True, output={"keywords": [], "matches": []})

        context_lines = max(0, int(context_lines or 0))
        max_results = max(1, int(max_results or 10))
        lines = text.splitlines() or [text]
        matches: list[dict[str, object]] = []

        for index, line in enumerate(lines):
            lowered_line = line.lower()
            for keyword in normalized_keywords:
                if keyword.lower() not in lowered_line:
                    continue

                start = max(0, index - context_lines)
                end = min(len(lines), index + context_lines + 1)
                matches.append(
                    {
                        "keyword": keyword,
                        "line_no": index + 1,
                        "line": line.strip(),
                        "context_before": [item.strip() for item in lines[start:index] if item.strip()],
                        "context_after": [item.strip() for item in lines[index + 1 : end] if item.strip()],
                    }
                )
                break

            if len(matches) >= max_results:
                break

        return ToolResult(
            success=True,
            output={
                "keywords": normalized_keywords,
                "matches": matches,
            },
        )


def _normalize_keywords(keywords: list[str]) -> list[str]:
    """清理关键词列表，去重并保持原顺序。"""
    result: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        cleaned = str(keyword).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result
