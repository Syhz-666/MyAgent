"""报告构建工具。

第三阶段 3A 中，报告标题和章节结构由 task 与 analysis.sections 动态决定。
"""

from __future__ import annotations

from typing import Any, Callable

from .base_tool import BaseTool, ToolResult


class BuildReportTool(BaseTool):
    """根据结构化分析结果生成 Markdown 报告。"""

    name = "build_report"
    description = "根据结构化分析结果、用户任务和执行步骤生成 Markdown 报告。"

    def __init__(self, report_builder: Callable[..., str]) -> None:
        self.report_builder = report_builder

    def run(self, analysis: dict[str, Any], steps: list[Any], task: str = "") -> ToolResult:
        """构建 Markdown 报告。"""
        if not analysis:
            return ToolResult(success=False, error="缺少 analysis，无法构建报告")

        try:
            report = self.report_builder(analysis, steps, task)
        except TypeError:
            # 兼容旧版只接收 analysis 和 steps 的 report_builder。
            report = self.report_builder(analysis, steps)

        return ToolResult(success=True, output=report)
