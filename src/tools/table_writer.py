"""表格写入工具。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .base_tool import BaseTool, ToolResult


class TableWriter(BaseTool):
    """将清洗后的表格数据写入 CSV 文件。"""

    name = "table_writer"
    description = "将清洗后的表格数据写入本地 CSV 文件。"

    def run(self, table_data: dict[str, Any], path: str, encoding: str = "utf-8-sig") -> ToolResult:
        """写入清洗后的 CSV。"""
        if not isinstance(table_data, dict):
            return ToolResult(success=False, error="缺少有效 table_data，无法写入表格")
        if not path:
            return ToolResult(success=False, error="缺少 path，无法写入表格")

        output_path = Path(path)
        if output_path.suffix.lower() != ".csv":
            output_path = output_path.with_suffix(".csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        columns = list(table_data.get("columns", []) or [])
        rows = list(table_data.get("rows", []) or [])
        try:
            with output_path.open("w", encoding=encoding, newline="") as file:
                writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        except Exception as exc:
            return ToolResult(success=False, error=f"写入 CSV 失败：{exc}")

        return ToolResult(success=True, output=str(output_path))
