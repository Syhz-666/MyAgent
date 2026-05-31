"""表格处理工具通用辅助函数。"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

NULL_VALUES = {"", "none", "null", "nan", "na", "n/a", "-", "--"}
BOOLEAN_TRUE_VALUES = {"是", "y", "yes", "true", "1", "对", "已", "已处理"}
BOOLEAN_FALSE_VALUES = {"否", "n", "no", "false", "0", "错", "未", "未处理"}
BOOLEAN_VALUES = BOOLEAN_TRUE_VALUES | BOOLEAN_FALSE_VALUES


def clean_cell(value: Any) -> str:
    """清理单元格为字符串，去掉首尾空白和常见 BOM。"""
    if value is None:
        return ""
    return str(value).replace("\ufeff", "").strip()


def is_empty_value(value: Any) -> bool:
    """判断值是否为空。"""
    text = clean_cell(value)
    return text.lower() in NULL_VALUES


def normalize_header(value: Any, index: int) -> str:
    """标准化表头。"""
    header = clean_cell(value).replace("\n", " ").replace("\r", " ").strip()
    if not header or header.lower().startswith("unnamed"):
        return f"未命名列{index}"
    return re.sub(r"\s+", " ", header)


def dedupe_headers(headers: list[str]) -> list[str]:
    """处理重复表头。"""
    result: list[str] = []
    counts: dict[str, int] = {}
    for header in headers:
        count = counts.get(header, 0)
        if count == 0:
            result.append(header)
        else:
            result.append(f"{header}_{count + 1}")
        counts[header] = count + 1
    return result


def parse_number(value: Any) -> float | None:
    """解析金额/数字文本。"""
    text = clean_cell(value)
    if is_empty_value(text):
        return None
    normalized = text.replace(",", "").replace("￥", "").replace("$", "").replace("元", "").replace(" ", "")
    try:
        return float(normalized)
    except ValueError:
        return None


def parse_date(value: Any) -> datetime | None:
    """解析常见日期格式。"""
    text = clean_cell(value)
    if is_empty_value(text):
        return None
    normalized = text.replace("年", "-").replace("月", "-").replace("日", "").replace("/", "-").replace(".", "-")
    normalized = re.sub(r"\s+", "", normalized)
    formats = ["%Y-%m-%d", "%Y-%m-%d%H:%M:%S", "%Y-%m", "%m-%d-%Y", "%d-%m-%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None


def is_boolean_value(value: Any) -> bool:
    """判断是否是常见布尔枚举值。"""
    text = clean_cell(value).lower()
    return text in BOOLEAN_VALUES


def infer_column_type(values: list[Any], threshold: float = 0.9) -> str:
    """根据非空值推断列类型。"""
    non_empty = [value for value in values if not is_empty_value(value)]
    if not non_empty:
        return "text"

    total = len(non_empty)
    boolean_count = sum(1 for value in non_empty if is_boolean_value(value))
    if boolean_count / total >= threshold:
        return "boolean"

    number_count = sum(1 for value in non_empty if parse_number(value) is not None)
    if number_count / total >= threshold:
        return "number"

    date_count = sum(1 for value in non_empty if parse_date(value) is not None)
    if date_count / total >= threshold:
        return "date"

    return "text"


def row_signature(row: dict[str, Any], columns: list[str]) -> tuple[str, ...]:
    """生成行去重签名。"""
    return tuple(clean_cell(row.get(column, "")) for column in columns)


def is_empty_row(row: dict[str, Any], columns: list[str]) -> bool:
    """判断是否为空行。"""
    return all(is_empty_value(row.get(column, "")) for column in columns)
