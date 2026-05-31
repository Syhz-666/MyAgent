"""LLM 客户端协议。"""

from __future__ import annotations

from typing import Any, Protocol


class LLMClient(Protocol):
    """文本处理 LLM 客户端协议。"""

    def analyze_text(
        self,
        text: str,
        task: str,
        search_results: dict[str, Any] | None = None,
        extract_action_items: bool = False,
    ) -> dict[str, Any]:
        """根据用户任务分析文本并返回统一结构化结果。"""
        ...

    def plan_text_processing(self, task: str, tool_descriptions: list[dict[str, Any]]) -> dict[str, Any]:
        """根据用户任务和工具描述生成结构化计划。"""
        ...
