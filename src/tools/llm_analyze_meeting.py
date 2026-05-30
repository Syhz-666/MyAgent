"""LLM 会议分析工具。

把 LLM 对会议内容的分析包装为 BaseTool，与其他工具统一注册到 Executor。
"""

from __future__ import annotations

from typing import Any, Protocol

from .base_tool import BaseTool, ToolResult


class LLMClient(Protocol):
    """LLM 客户端协议，与 agent.py 中的 LLMClient 保持一致。"""

    def analyze_meeting(self, meeting_text: str) -> dict[str, Any]:
        ...


class LLMAnalyzeMeetingTool(BaseTool):
    """调用 LLM 从会议记录中提取结构化信息。"""

    name = "llm_analyze_meeting"
    description = "分析会议记录文本，提取会议概要、关键结论、行动项、风险和待确认问题。"

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def run(self, meeting_text: str) -> ToolResult:
        if not meeting_text:
            return ToolResult(success=False, error="缺少 meeting_text，无法分析会议记录")
        analysis = self.llm_client.analyze_meeting(meeting_text)
        if not analysis:
            return ToolResult(success=False, error="LLM 未返回有效结构化结果")
        return ToolResult(success=True, output=analysis)
