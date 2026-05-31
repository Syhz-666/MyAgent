"""表格质量诊断工具。"""

from __future__ import annotations

from typing import Any

from .base_tool import BaseTool, ToolResult
from .table_utils import is_empty_row, is_empty_value, parse_date, parse_number, row_signature


class TableProfiler(BaseTool):
    """分析表格数据质量问题。"""

    name = "table_profiler"
    description = "分析表格数据质量问题，输出结构化的诊断结果。"

    def run(self, table_data: dict[str, Any]) -> ToolResult:
        """执行表格质量诊断。"""
        if not isinstance(table_data, dict):
            return ToolResult(success=False, error="缺少有效 table_data，无法诊断表格")

        columns = list(table_data.get("columns", []) or [])
        rows = list(table_data.get("rows", []) or [])
        column_types = dict(table_data.get("column_types", {}) or {})
        issues: list[dict[str, Any]] = []

        empty_row_indices = [index + 1 for index, row in enumerate(rows) if is_empty_row(row, columns)]
        if empty_row_indices:
            issues.append(_issue("empty_rows", "low", len(empty_row_indices), empty_row_indices))

        empty_columns = [column for column in columns if all(is_empty_value(row.get(column, "")) for row in rows)]
        for column in empty_columns:
            issues.append(
                {
                    "type": "empty_columns",
                    "column": column,
                    "severity": "low",
                    "count": 1,
                    "sample_indices": [],
                }
            )

        seen: dict[tuple[str, ...], int] = {}
        duplicate_indices: list[int] = []
        for index, row in enumerate(rows, start=1):
            if is_empty_row(row, columns):
                continue
            signature = row_signature(row, columns)
            if signature in seen:
                duplicate_indices.append(index)
            else:
                seen[signature] = index
        if duplicate_indices:
            issues.append(_issue("duplicate_rows", "medium", len(duplicate_indices), duplicate_indices))

        column_profiles: list[dict[str, Any]] = []
        for column in columns:
            values = [row.get(column, "") for row in rows]
            non_null_values = [value for value in values if not is_empty_value(value)]
            null_indices = [index + 1 for index, value in enumerate(values) if is_empty_value(value)]
            column_profiles.append(
                {
                    "column": column,
                    "inferred_type": column_types.get(column, "text"),
                    "non_null_count": len(non_null_values),
                    "null_count": len(null_indices),
                    "unique_count": len({str(value) for value in non_null_values}),
                }
            )
            if null_indices:
                issues.append(
                    {
                        "type": "missing_values",
                        "column": column,
                        "severity": "medium",
                        "count": len(null_indices),
                        "sample_indices": null_indices[:10],
                    }
                )

            mismatches = _type_mismatch_indices(values, column_types.get(column, "text"))
            if mismatches:
                issues.append(
                    {
                        "type": "type_mismatch",
                        "column": column,
                        "expected_type": column_types.get(column, "text"),
                        "severity": "high",
                        "count": len(mismatches),
                        "sample_indices": mismatches[:10],
                    }
                )

            irregular_indices = [
                index + 1
                for index, value in enumerate(values)
                if isinstance(value, str) and value and (value != value.strip() or "\n" in value or "\r" in value)
            ]
            if irregular_indices:
                issues.append(
                    {
                        "type": "irregular_text",
                        "column": column,
                        "severity": "low",
                        "count": len(irregular_indices),
                        "sample_indices": irregular_indices[:10],
                    }
                )

        header_anomalies = [column for column in columns if column.startswith("未命名列") or "\n" in column or "\r" in column]
        if header_anomalies:
            issues.append(
                {
                    "type": "header_anomaly",
                    "column": ", ".join(header_anomalies[:5]),
                    "severity": "medium",
                    "count": len(header_anomalies),
                    "sample_indices": [],
                }
            )

        return ToolResult(
            success=True,
            output={
                "row_count": len(rows),
                "column_count": len(columns),
                "issues": issues,
                "column_profiles": column_profiles,
            },
        )


def _issue(issue_type: str, severity: str, count: int, sample_indices: list[int]) -> dict[str, Any]:
    """构造通用问题对象。"""
    return {
        "type": issue_type,
        "severity": severity,
        "count": count,
        "sample_indices": sample_indices[:10],
    }


def _type_mismatch_indices(values: list[Any], expected_type: str) -> list[int]:
    """保守识别类型异常。"""
    if expected_type not in {"number", "date", "boolean"}:
        return []

    mismatches: list[int] = []
    for index, value in enumerate(values, start=1):
        if is_empty_value(value):
            continue
        if expected_type == "number" and parse_number(value) is None:
            mismatches.append(index)
        elif expected_type == "date" and parse_date(value) is None:
            mismatches.append(index)
        elif expected_type == "boolean" and str(value).strip().lower() not in {"是", "否", "y", "n", "yes", "no", "true", "false", "1", "0", "对", "错", "已", "未", "已处理", "未处理"}:
            mismatches.append(index)
    return mismatches
