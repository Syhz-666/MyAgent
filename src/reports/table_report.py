"""表格清洗报告生成。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_table_cleaning_report(
    source_path: str,
    table_data: dict[str, Any],
    profile: dict[str, Any],
    cleaner_output: dict[str, Any],
    steps: list[Any] | None = None,
    cleaned_path: str = "",
) -> str:
    """生成 Markdown 表格清洗报告。"""
    source_name = Path(source_path).name if source_path else "未知文件"
    original_row_count = table_data.get("row_count", 0) if isinstance(table_data, dict) else 0
    original_column_count = len(table_data.get("columns", []) or []) if isinstance(table_data, dict) else 0
    cleaned_row_count = cleaner_output.get("row_count", 0) if isinstance(cleaner_output, dict) else 0
    issues = profile.get("issues", []) if isinstance(profile, dict) else []
    operations = cleaner_output.get("operations", []) if isinstance(cleaner_output, dict) else []
    warnings = cleaner_output.get("warnings", []) if isinstance(cleaner_output, dict) else []

    lines: list[str] = []
    lines.append(f"# 表格清洗报告：{source_name}")
    lines.append("")

    lines.append("## 输入文件")
    lines.append("")
    lines.append(f"- 文件名：{source_name}")
    lines.append(f"- 原始行数：{original_row_count}")
    lines.append(f"- 原始列数：{original_column_count}")
    lines.append(f"- 清洗后行数：{cleaned_row_count}")
    lines.append("")

    lines.append("## 识别到的问题")
    lines.append("")
    if issues:
        lines.append("| 问题类型 | 字段 | 数量 | 严重程度 | 示例行 |")
        lines.append("|----------|------|------|----------|--------|")
        for issue in issues:
            lines.append(
                "| {type} | {column} | {count} | {severity} | {samples} |".format(
                    type=_escape(str(issue.get("type", "待确认"))),
                    column=_escape(str(issue.get("column", "-"))),
                    count=_escape(str(issue.get("count", 0))),
                    severity=_escape(str(issue.get("severity", "待确认"))),
                    samples=_escape(", ".join(str(index) for index in issue.get("sample_indices", []) or []) or "-"),
                )
            )
    else:
        lines.append("- 未识别到明显数据质量问题。")
    lines.append("")

    lines.append("## 已执行的清洗操作")
    lines.append("")
    if operations:
        for operation in operations:
            lines.append(f"- {_format_operation(operation)}")
    else:
        lines.append("- 未执行会改变数据内容的清洗操作。")
    lines.append("")

    lines.append("## 需要人工确认")
    lines.append("")
    if warnings:
        for warning in warnings:
            column = warning.get("column", "-")
            rows = ", ".join(str(index) for index in warning.get("row_indices", []) or []) or "-"
            note = warning.get("note", "需人工确认")
            lines.append(f"- `{column}`：{note}，示例行：{rows}")
    else:
        lines.append("- 暂无需要人工确认的异常值。")
    lines.append("")

    lines.append("## 输出文件")
    lines.append("")
    lines.append(f"- 清洗后表格：{cleaned_path or '待确认'}")
    lines.append("- 清洗报告：本文件")
    lines.append("")

    if steps:
        lines.append("## 执行过程")
        lines.append("")
        lines.append("| 步骤 | 思考 | 动作 | 观察结果 | 状态 |")
        lines.append("|------|------|------|----------|------|")
        for step in steps:
            lines.append(
                "| {step} | {thought} | {action} | {observation} | {status} |".format(
                    step=getattr(step, "step", ""),
                    thought=_escape(str(getattr(step, "thought", ""))),
                    action=_escape(str(getattr(step, "action", ""))),
                    observation=_escape(str(getattr(step, "observation", ""))),
                    status=_escape(str(getattr(step, "status", ""))),
                )
            )
        lines.append("")

    return "\n".join(lines)


def _format_operation(operation: dict[str, Any]) -> str:
    """格式化清洗操作说明。"""
    operation_type = operation.get("type", "unknown")
    if "affected_count" in operation:
        return f"{operation_type}：{operation.get('affected_count', 0)} 项"
    if "affected_cells" in operation:
        return f"{operation_type}：{operation.get('affected_cells', 0)} 个单元格"
    if "affected_columns" in operation:
        return f"{operation_type}：{operation.get('affected_columns', 0)} 个字段"
    return str(operation)


def _escape(value: str) -> str:
    """转义 Markdown 表格内容。"""
    return value.replace("|", "\\|").replace("\n", " ")
