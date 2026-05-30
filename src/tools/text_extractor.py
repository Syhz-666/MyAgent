"""通用文本分析工具。

第三阶段 3A 中，TextExtractor 不再绑定“会议记录”场景，
而是根据用户传入的 task 动态调整分析目标，并返回统一结构化结果。
"""

from __future__ import annotations

from typing import Any, Protocol

from .base_tool import BaseTool, ToolResult


class LLMClient(Protocol):
    """文本分析 LLM 客户端协议。"""

    def analyze_text(self, text: str, task: str) -> dict[str, Any]:
        """根据用户任务分析文本，并返回统一结构化结果。"""
        ...


class TextExtractor(BaseTool):
    """根据用户任务从文本中提取结构化信息。"""

    name = "text_extractor"
    description = "根据用户指定的任务，从文本中提取结构化信息。"

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def run(self, text: str, task: str = "") -> ToolResult:
        """执行文本分析。

        Args:
            text: 待分析的原文文本。
            task: 用户指定的分析目标。
        """
        if not text:
            return ToolResult(success=False, error="缺少 text，无法分析文本")

        task = task or "请分析这份文本并提取关键信息。"

        if hasattr(self.llm_client, "analyze_text"):
            analysis = self.llm_client.analyze_text(text, task)
        elif hasattr(self.llm_client, "analyze_meeting"):
            # 向后兼容第二阶段的 LLMClient。
            analysis = self.llm_client.analyze_meeting(text)  # type: ignore[attr-defined]
        else:
            return ToolResult(success=False, error="LLM 客户端缺少 analyze_text 方法")

        if not isinstance(analysis, dict) or not analysis:
            return ToolResult(success=False, error="LLM 未返回有效结构化结果")

        return ToolResult(success=True, output=analysis)
