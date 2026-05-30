"""会议分析工具兼容层。

第三阶段 3A 已将通用分析能力迁移到 TextExtractor。
本模块保留 LLMAnalyzeMeetingTool，避免旧代码 import 失效。
"""

from __future__ import annotations

from .base_tool import ToolResult
from .text_extractor import LLMClient, TextExtractor


class LLMAnalyzeMeetingTool(TextExtractor):
    """兼容第二阶段的会议分析工具名。"""

    name = "llm_analyze_meeting"
    description = "兼容旧版会议记录分析工具，内部复用 TextExtractor。"

    def __init__(self, llm_client: LLMClient) -> None:
        super().__init__(llm_client)

    def run(
        self,
        meeting_text: str = "",
        text: str = "",
        task: str = "请整理这份会议记录，提取会议概要、关键结论、行动项和风险点。",
    ) -> ToolResult:
        """兼容旧参数 meeting_text，同时支持新参数 text。"""
        return super().run(text=text or meeting_text, task=task)
