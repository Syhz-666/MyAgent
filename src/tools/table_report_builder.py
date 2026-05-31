"""表格清洗报告工具。"""

from __future__ import annotations

from typing import Any

try:
    from ..reports.table_report import build_table_cleaning_report
except ImportError:  # pragma: no cover - 支持直接导入
    from reports.table_report import build_table_cleaning_report

from .base_tool import BaseTool, ToolResult


class TableReportBuilder(BaseTool):
    """根据清洗结果和操作记录生成 Markdown 清洗报告。"""

    name = "table_report_builder"
    description = "根据清洗结果和操作记录生成 Markdown 清洗报告。"

    def run(
        self,
        table_data: dict[str, Any],
        profile: dict[str, Any],
        cleaner_output: dict[str, Any],
        steps: list[Any] | None = None,
        cleaned_path: str = "",
    ) -> ToolResult:
        """生成表格清洗报告。"""
        if not isinstance(table_data, dict):
            return ToolResult(success=False, error="缺少 table_data，无法生成表格清洗报告")
        if not isinstance(profile, dict):
            return ToolResult(success=False, error="缺少 profile，无法生成表格清洗报告")
        if not isinstance(cleaner_output, dict):
            return ToolResult(success=False, error="缺少 cleaner_output，无法生成表格清洗报告")

        report = build_table_cleaning_report(
            source_path=str(table_data.get("source_path", "")),
            table_data=table_data,
            profile=profile,
            cleaner_output=cleaner_output,
            steps=steps or [],
            cleaned_path=cleaned_path,
        )
        return ToolResult(success=True, output=report)
