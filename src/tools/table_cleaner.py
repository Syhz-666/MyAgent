"""表格清洗工具。"""

from __future__ import annotations

from typing import Any

from .base_tool import BaseTool, ToolResult
from .table_utils import clean_cell, is_empty_row, is_empty_value, parse_number, row_signature


class TableCleaner(BaseTool):
    """根据诊断结果执行安全清洗。"""

    name = "table_cleaner"
    description = "根据诊断结果执行安全清洗，不删除可能有业务意义的数据。"

    def run(self, table_data: dict[str, Any], profile: dict[str, Any]) -> ToolResult:
        """执行表格清洗。"""
        if not isinstance(table_data, dict):
            return ToolResult(success=False, error="缺少有效 table_data，无法清洗表格")

        columns = list(table_data.get("columns", []) or [])
        rows = [dict(row) for row in table_data.get("rows", []) or []]
        column_types = dict(table_data.get("column_types", {}) or {})
        operations: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        original_columns = list(columns)
        kept_columns = [column for column in columns if not all(is_empty_value(row.get(column, "")) for row in rows)]
        removed_columns = [column for column in columns if column not in kept_columns]
        if removed_columns:
            operations.append({"type": "drop_empty_columns", "affected_count": len(removed_columns), "columns": removed_columns})
        columns = kept_columns
        column_types = {column: column_types.get(column, "text") for column in columns}
        rows = [{column: row.get(column, "") for column in columns} for row in rows]

        non_empty_rows = [row for row in rows if not is_empty_row(row, columns)]
        dropped_empty_rows = len(rows) - len(non_empty_rows)
        if dropped_empty_rows:
            operations.append({"type": "drop_empty_rows", "affected_count": dropped_empty_rows})
        rows = non_empty_rows

        seen: set[tuple[str, ...]] = set()
        deduped_rows: list[dict[str, Any]] = []
        duplicate_count = 0
        for row in rows:
            signature = row_signature(row, columns)
            if signature in seen:
                duplicate_count += 1
                continue
            seen.add(signature)
            deduped_rows.append(row)
        if duplicate_count:
            operations.append({"type": "drop_duplicates", "affected_count": duplicate_count})
        rows = deduped_rows

        trimmed_cells = 0
        missing_cells = 0
        for row in rows:
            for column in columns:
                old_value = row.get(column, "")
                cleaned = clean_cell(old_value).replace("\r", " ").replace("\n", " ").strip()
                if cleaned != old_value:
                    trimmed_cells += 1
                if is_empty_value(cleaned):
                    row[column] = "待确认"
                    missing_cells += 1
                else:
                    row[column] = cleaned
        if trimmed_cells:
            operations.append({"type": "trim_text", "affected_cells": trimmed_cells})
        if missing_cells:
            operations.append({"type": "mark_missing", "affected_cells": missing_cells, "value": "待确认"})

        affected_headers = sum(1 for column in original_columns if column.startswith("未命名列"))
        if affected_headers:
            operations.append({"type": "normalize_headers", "affected_columns": affected_headers})

        for column in columns:
            if column_types.get(column) != "number":
                continue
            negative_indices: list[int] = []
            for index, row in enumerate(rows, start=1):
                number = parse_number(row.get(column, ""))
                if number is not None and number < 0:
                    negative_indices.append(index)
            if negative_indices:
                warnings.append(
                    {
                        "type": "negative_value",
                        "column": column,
                        "row_indices": negative_indices[:20],
                        "note": "数值为负数，已保留未删除",
                    }
                )

        output = {
            "columns": columns,
            "column_types": column_types,
            "rows": rows,
            "row_count": len(rows),
            "source_path": table_data.get("source_path", ""),
            "operations": operations,
            "warnings": warnings,
            "profile": profile,
        }
        return ToolResult(success=True, output=output)
