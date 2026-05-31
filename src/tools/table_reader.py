"""CSV 表格读取工具。"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from .base_tool import BaseTool, ToolResult
from .table_utils import clean_cell, dedupe_headers, infer_column_type, normalize_header


class TableReader(BaseTool):
    """读取 CSV 文件，转换为统一表格结构。"""

    name = "table_reader"
    description = "读取 CSV 文件，转换为统一的 TableData 结构，并自动推断列类型。"

    def run(self, path: str, encoding: str = "utf-8") -> ToolResult:
        """读取 CSV 文件。"""
        if not path:
            return ToolResult(success=False, error="缺少 path，无法读取表格")

        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            return ToolResult(success=False, error=f"表格文件不存在：{path}")

        if file_path.suffix.lower() != ".csv":
            return ToolResult(success=False, error="当前阶段仅支持 CSV 文件")

        try:
            rows = _read_csv(file_path, encoding)
        except UnicodeDecodeError:
            try:
                rows = _read_csv(file_path, "utf-8-sig")
            except UnicodeDecodeError:
                rows = _read_csv(file_path, "gbk")
        except Exception as exc:
            return ToolResult(success=False, error=f"读取 CSV 失败：{exc}")

        if not rows:
            return ToolResult(success=False, error="CSV 文件为空，无法处理")

        rows = _normalize_rows(rows)
        raw_headers = rows[0]
        headers = dedupe_headers([normalize_header(value, index + 1) for index, value in enumerate(raw_headers)])
        data_rows: list[dict[str, str]] = []
        for raw_row in rows[1:]:
            padded = list(raw_row) + [""] * max(0, len(headers) - len(raw_row))
            row = {column: clean_cell(padded[index]) if index < len(padded) else "" for index, column in enumerate(headers)}
            data_rows.append(row)

        column_types = {
            column: infer_column_type([row.get(column, "") for row in data_rows])
            for column in headers
        }

        return ToolResult(
            success=True,
            output={
                "columns": headers,
                "column_types": column_types,
                "rows": data_rows,
                "row_count": len(data_rows),
                "source_path": str(file_path),
            },
        )


def _read_csv(path: Path, encoding: str) -> list[list[str]]:
    """读取 CSV 为二维列表。"""
    with path.open("r", encoding=encoding, newline="") as file:
        sample = file.read(4096)
        file.seek(0)
        dialect = _detect_dialect(sample)
        return [row for row in csv.reader(file, dialect)]


def _detect_dialect(sample: str) -> type[csv.Dialect] | csv.Dialect:
    """安全识别 CSV 方言，避免 Sniffer 将换行符误判为分隔符。"""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        if dialect.delimiter in {",", "\t", ";", "|"}:
            return dialect
    except csv.Error:
        pass
    return csv.excel


def _normalize_rows(rows: list[list[str]]) -> list[list[str]]:
    """修正因未加引号金额逗号导致的行长度不一致问题。"""
    if len(rows) <= 1:
        return rows

    header_len = len(rows[0])
    if header_len <= 0:
        return rows

    normalized = [rows[0]]
    for row in rows[1:]:
        normalized.append(_normalize_row_length(row, header_len))
    return normalized


def _normalize_row_length(row: list[str], expected_len: int) -> list[str]:
    """将过长行中的疑似数字片段合并回前一列，保持列数稳定。"""
    if len(row) <= expected_len:
        return row

    cells = list(row)
    while len(cells) > expected_len:
        merge_index = _find_numeric_fragment_index(cells)
        if merge_index is None:
            merge_index = len(cells) - 1
        cells[merge_index - 1] = f"{cells[merge_index - 1]},{cells[merge_index]}"
        del cells[merge_index]
    return cells


def _find_numeric_fragment_index(cells: list[str]) -> int | None:
    """查找被金额千分位逗号拆开的后半段数字。"""
    for index in range(1, len(cells)):
        current = clean_cell(cells[index]).replace(" ", "")
        previous = clean_cell(cells[index - 1]).replace(" ", "")
        if re.fullmatch(r"\d{3}(?:\.\d+)?", current) and re.search(r"[￥$]?[-+]?\d+$", previous):
            return index
    return None
