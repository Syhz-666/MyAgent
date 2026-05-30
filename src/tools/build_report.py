"""报告构建工具。

把 Markdown 报告生成包装为 BaseTool，与其他工具统一注册到 Executor。
"""

from __future__ import annotations

from typing import Any, Callable

from .base_tool import BaseTool, ToolResult


class BuildReportTool(BaseTool):
    """根据 LLM 分析结果生成 Markdown 报告。"""

    name = "build_report"
    description = "根据结构化分析结果和执行步骤生成 Markdown 格式的会议整理报告。"

    def __init__(self, report_builder: Callable[..., str]) -> None:
        self.report_builder = report_builder

    def run(self, analysis: dict[str, Any], steps: list[Any]) -> ToolResult:
        if not analysis:
            return ToolResult(success=False, error="缺少 analysis，无法构建报告")
        report = self.report_builder(analysis, steps)
        return ToolResult(success=True, output=report)
